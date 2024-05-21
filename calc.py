from sympy import Symbol
from zq_tools.zq_logger import default_logger as logger


s = Symbol("s")
h = Symbol("h")
v = Symbol("v")
L = Symbol("L")
a = Symbol("a")
f = Symbol("f") # FLOPS of single GPU
d_k = d_v = h/a
d_ff = h*4
# logger.warn_root(f"Assume: d_k=d_v=h//a")
# logger.warn(f"Assume: d_ff=h*4")

num_params_head = v*h+(s+1)*h # Word Embedding + Pos Embedding
num_params_tail = v*h # Linear (bias=False)
num_params_single_layer = (h*h+h)*3+(a*d_v*h+h) + (h+h) + (h*d_ff+h)+(h*d_ff+d_ff) + (h+h)  # (Q,K,V Attention) + Linear(h->h) + LayerNorm + Linear(h->4h) + Linear(4h->h) + LayerNorm
num_params_all = num_params_head+num_params_tail+num_params_single_layer*L

# =============== counting tflop===============
logger.debug_root(f"For FLOPs calculations, we only consider the matrix multiplications (GEMMs) which are the main contributors to the number of floating-point operations")
flops_head = 0 # Embedding costs little time
flops_per_transformer = 2*s*h*h*3 + 2*s*s*h + 2*s*s*h + 2*s*h*h + 2*s*h*(4*h) + 2*s*(4*h)*h # (Q,K,V Attention) + (Attention matrix computation QxK) + (Attention over values) + (Post attention linear projection) + (Linear h->4h) + Linear( 4h->h)
flops_selective_recomp_part = 4*s*s*h 
flops_selective_recomp_part2 = 2*s*h*h*3 + 2*s*s*h + 2*s*s*h + 2*s*h*h
flops_tail = 2*s*h*v
logger.debug_root(f"The backward pass requires double the number of FLOPs since we need to calculate the gradients with respect to both input and weight tensors")
flops_forward = flops_head + flops_per_transformer*L + flops_tail
flops_backward = flops_forward*2
model_flops = flops_forward + flops_backward
full_recomp_flops = flops_forward*2 + flops_backward

# activations in bytes (fp16)
activations_head = s*h
activations_per_transformer  = 34*s*h + 5*a*s*s
activations_tail = 4*s*h+4*s*v
activations_tail_fp32 = 8*s*h+4*s*v
activations_all = activations_head + activations_per_transformer*L + activations_tail
activations_selective_recomp_part = 5*a*s*s
activations_selective_recomp_part2 = 5*a*s*s + 23*s*h

select_level = 'nvidia'
if select_level == 'mha':
    flops_selective_recomp_part = flops_selective_recomp_part2
    activations_selective_recomp_part = activations_selective_recomp_part2
    




activation_checkpoint_nbytes_fp32 = s*h*4 + s*s*1
activation_checkpoint_nbytes_fp16 = s*h*2 + s*s*1
X_nbytes_fp32 = s*h*4
X_nbytes_fp16 = s*h*2
input_size = s*8 # dtype: Long


required_pcie_bandwidth = (activations_per_transformer-activations_selective_recomp_part)*f/(3*flops_per_transformer)

from typing import Dict
def get_substitute(config_path:str)->Dict[str,int]:
    with open(config_path, 'r') as f:
        content = f.read()
        import json
        j = json.loads(content)
        d_model = j['d_model']
        d_ff = j['d_ff']
        if d_model*4!=d_ff:
            logger.warn_root(f"All results are for d_ff==d_model*4, but d_ff={d_ff}!=d_model*4={d_model*4}")
        substitute = {
            s: j['seq_len'],
            h: j['d_model'],
            v: j['vocab_size'],
            L: j['n_layers'],
            a: j['n_heads']
        }
        return substitute

def get_training_flops(config_path:str, recompute_strategy="always")->float:
    if recompute_strategy in ["always", "full"]:
        flops = flops_forward + flops_backward + flops_forward
    elif recompute_strategy == "selective":
        flops = flops_forward + flops_backward + flops_selective_recomp_part
    elif recompute_strategy in ["never", "disable"]:
        flops = flops_forward + flops_backward
    else:
        raise Exception(f"Unknown recompute_strategy: {recompute_strategy}")
    logger.info_root(f"flops_forward={flops_forward}")
    logger.info_root(f"flops={flops}")
    subs=get_substitute(config_path)
    logger.info_root(f"subs={subs}")
    flops = flops.subs(subs)
    return flops

    

def generate_table(config_path:str=""):
    if config_path: 
        substitute = get_substitute(config_path)
        logger.info_root(f"subs={substitute}")
    else:
        logger.info_root(f"pure symbol")
    def format_num_params(n:Symbol):
        if not config_path: return n
        try: n=n.subs(substitute)
        except: pass
        return f"[{n} = {n/1e9:.2f} B= {n*4/1024/1024/1024:.2f} GiB]"
    def format_flops(n:Symbol):
        if not config_path: return n
        try: n=n.subs(substitute)
        except: pass
        return f"[{n} FLOPs={n/1e12:.2f} TeraFLOPs]"
    def format_tensor_size(n:Symbol):
        if not config_path: return n
        try: n=n.subs(substitute)
        except: pass
        return f"[{n} Bytes={n/1024/1024:.2f} MiB]"
    from prettytable import PrettyTable
    table = PrettyTable(["phase", "FLOPs/bs", "Weight(FP32)", "Input size/bs(FP16)", "Activations/bs(fp16)"])
    table.add_row(["Head", format_flops(flops_head), format_num_params(num_params_head), format_tensor_size(input_size), format_tensor_size(activations_head)])
    table.add_row(["Transformer", format_flops(flops_per_transformer), format_num_params(num_params_single_layer), format_tensor_size(activation_checkpoint_nbytes_fp16), format_tensor_size(activations_per_transformer)])
    table.add_row(["Selected Recomp Part", format_flops(flops_selective_recomp_part), "TBD", "TBD", format_tensor_size(activations_selective_recomp_part)])
    table.add_row(["Tail", format_flops(flops_tail), format_num_params(num_params_tail), format_tensor_size(activation_checkpoint_nbytes_fp16), format_tensor_size(activations_tail)])
    table.add_row(["Model", format_flops(flops_forward), format_num_params(num_params_all), format_tensor_size(input_size), format_tensor_size(activations_all)])
    table.add_row(["FW+BW", format_flops(model_flops), "N/A", "N/A", "N/A"])
    table.add_row(["FW+BW+(full)Recomp.", format_flops(full_recomp_flops), "N/A", "N/A", "N/A"])
    table.add_row(["FW+BW+(selective)Recomp.", format_flops(flops_forward+flops_backward+flops_selective_recomp_part*L), "N/A", "N/A", "N/A"])
    return str(table)

if __name__ == '__main__':
    config_path="gpt3.json"
    print(generate_table())
    print(generate_table(config_path))
    print(get_training_flops(config_path, "always"))
    subs=get_substitute(config_path)
    max_f = 65*0.3
    subs[f]=max_f*1e12
    print(f"required_pcie_bandwidth={required_pcie_bandwidth.subs(subs)/1024/1024/1024:.2f} GiB/s")


# zero-infinity analysis
