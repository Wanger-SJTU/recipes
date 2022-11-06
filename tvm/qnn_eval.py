#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 2021-09-22 8:20

import tvm
from tvm import te
import numpy as np
from tvm import relay

x_data = np.array((1, 153, 2, 178)).reshape((1, 4))
y_data = np.array((204, 178, 1, 8)).reshape((1, 4))

x_scale = y_scale = z_scale = 0.00784314
x_zero_point = y_zero_point = z_zero_point = 127


def dequant(data, scale, zp):
    return scale * (np.asarray(data) - zp)


def quant(data, scale, zp):
    z = np.around(data / scale + zp)
    q_min = np.iinfo(np.uint8).min
    q_max = np.iinfo(np.uint8).max
    return np.clip(z, q_min, q_max)


def mul_manually():
    return quant(
        dequant(x_data, x_scale, x_zero_point) * dequant(y_data, y_scale, y_zero_point),
        z_scale,
        z_zero_point,
    )


if __name__ == "__main__":
    x = relay.var("x", shape=(1, 4), dtype="uint8")
    y = relay.var("y", shape=(1, 4), dtype="uint8")
    z = relay.qnn.op.mul(
        lhs=x,
        rhs=y,
        lhs_scale=relay.const(x_scale, "float32"),
        lhs_zero_point=relay.const(x_zero_point, "int32"),
        rhs_scale=relay.const(y_scale, "float32"),
        rhs_zero_point=relay.const(y_zero_point, "int32"),
        output_scale=relay.const(z_scale, "float32"),
        output_zero_point=relay.const(z_zero_point, "int32"),
    )

    func = relay.Function([x, y], z)
    mod = tvm.IRModule.from_expr(func)
    print("----------before qnn transform----------")
    print(mod)
    mod = relay.transform.InferType()(mod)
    mod = relay.qnn.transform.CanonicalizeOps()(mod)
    print("----------after qnn transform----------")
    print(mod)
    func = mod["main"]

    intrp = relay.create_executor("graph", device=tvm.cpu(0), target="llvm")
    op_res = intrp.evaluate(func)(x_data, y_data)

    print("----------")
    print(op_res.numpy())
    print("----------")
    golden = mul_manually()
    print(golden.astype("uint8"))