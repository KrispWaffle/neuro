from __future__ import annotations
from enum import Enum, auto
from dataclasses import dataclass, field
import numpy as np
import itertools
from typing import Callable, Dict
class OpType(Enum):
    CONST = auto()      
    LOAD  = auto()     
    STORE  = auto()     
    ADD = auto()
    SUB = auto()
    MUL  = auto()
    RELU = auto()
    MATMUL = auto()      
_id_gen = itertools.count()
OP_MAP = {
    OpType.ADD:    lambda inputs: inputs[0] + inputs[1],
    OpType.SUB:    lambda inputs: inputs[0] - inputs[1],
    OpType.MUL:    lambda inputs: inputs[0] * inputs[1],
    OpType.MATMUL: lambda inputs: np.matmul(inputs[0], inputs[1]),
    OpType.RELU:   lambda inputs: np.maximum(inputs[0], 0),
   
}
@dataclass(frozen=True, slots=True, eq=False)
class Instr:
    op: OpType
    dtype: np.dtype
    src: tuple["Instr", ...] = field(default_factory=tuple) 
    arg:     object | None = None
    id:      int = field(default_factory=lambda: next(_id_gen))
    def __str__(self): 
        src_ids = [s.id for s in self.src ]
        if(not src_ids):
            return f"{self.id:03d}: {self.op.name:<6} arg={self.arg}"
        else:
            return f"{self.id:03d}: {self.op.name:<6} src={src_ids} arg={self.arg}"   
class Program(list[Instr]):
    def append_and_return(self, instr: Instr) -> Instr:
        super().append(instr)
        return instr
def find_sinks(comp: Program) -> list[Instr]:
    used = {s for instr in comp for s in instr.src}
    return [instr for instr in comp if instr not in used]  
def add_stores(comp: Program, base="out"):
    # ignore existing STOREs when looking for new sinks
    sinks = [i for i in find_sinks(comp) if i.op is not OpType.STORE]
    for k, instr in enumerate(sinks):
        comp.append_and_return(
            Instr(OpType.STORE, instr.dtype,
                  src=(instr,), arg=f"{base}{k if len(sinks)>1 else ''}"))
EmitFn = Callable[[Instr],str]
def _emit_const(i: Instr) -> str:
    name = f"tmp{i.id}"
    return f"float {name} = {i.arg};"

def _emit_binary(op_symbol: str) -> EmitFn:
    def fn(i: Instr) -> str:
        a, b = i.src
        return f"float tmp{i.id} = tmp{a.id} {op_symbol} tmp{b.id};"
    return fn
RULES: Dict[OpType, EmitFn] = {
    OpType.CONST : _emit_const, 
    OpType.ADD   : _emit_binary('+'),
    OpType.SUB   : _emit_binary('-'),
    OpType.MUL   : _emit_binary('*'),
    OpType.RELU  : lambda i: (
        f"float tmp{i.id} = fmaxf(0.f, tmp{i.src[0].id});"
    ),
    OpType.MATMUL: lambda i: "// TODO: matmul kernel call",
    OpType.LOAD  : lambda i: f"float tmp{i.id} = {i.arg}[idx];",
    OpType.STORE : lambda i: f"{i.arg}[idx] = tmp{i.src[0].id};",
}
def print_comp(comp):
    for i in comp:
        print(i)
        
def realize(program: Program):
   sorted =  topo_sort(program)
   print("called realize")
   buffer:dict[float, np.array] = {}
   for i in sorted:
       if i  == OpType.CONST:
           buffer[i.id] = np.array(i.arg, dtype=i.dtype)
       else:
           operation = OP_MAP[i.op]
           
           input_data = [buffer[s.id] for s in i.src]
           buffer[i.id] = operation(input_data)
           
   return buffer[sorted[-1].id]
    
            
def topo_sort(Program: Program) -> list[Instr]:
    indeg = {i: 0 for i in Program}
    for i in Program:
        for s in i.src:
            indeg[s] += 1
    ready, order = [i for i,d in indeg.items() if d==0], []
    while ready:
        n = ready.pop()
        order.append(n)
        for m in n.src:        
            indeg[m] -= 1
            if indeg[m] == 0:
                ready.append(m)
    print(order[::-1])
    return order[::-1] 
def emit_cuda_kernel(comp: Program, kern_name="kernel"):
    body = []
    add_stores(comp=comp )

    for instr in topo_sort(comp):
        body.append("    " + RULES[instr.op](instr))
    body_str = "\n".join(body)

    return f"""
__global__ void {kern_name}(int N, const float* a, const float* b, float* out) {{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= N) return;
{body_str}
}}
""".strip()