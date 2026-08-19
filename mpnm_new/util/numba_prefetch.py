import numba as nb
from numba import types
from llvmlite import ir


@nb.extending.intrinsic
def _prefetch_ptr_write(typingctx, ptr):
    if isinstance(ptr, types.Integer):

        def codegen(context, builder, sig, args):
            int8_p = ir.IntType(8).as_pointer()
            # llvm.prefetch (is_write=1, locality=0, cache_type=1)
            fnty = ir.FunctionType(
                ir.VoidType(), [int8_p, ir.IntType(32), ir.IntType(32), ir.IntType(32)]
            )
            prefetch = builder.module.declare_intrinsic("llvm.prefetch", [], fnty)
            addr = builder.inttoptr(args[0], int8_p)
            # (addr, is_write=1, locality=0, cache_type=1)
            builder.call(
                prefetch,
                [
                    addr,
                    ir.Constant(ir.IntType(32), 1),
                    ir.Constant(ir.IntType(32), 3),
                    ir.Constant(ir.IntType(32), 1),
                ],
            )

        return types.void(types.uintp), codegen
