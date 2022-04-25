using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System;

[System.Serializable]
public struct Parameter
{
    public double h;

    public double r0;
    public double h0;
    public double beta01;
    public double beta02;
    public double beta03;
    public double b0;
    public double beta1;
    public double beta2;
    public double a1;
    public double a2;
}

public class ADRC : MonoBehaviour
{
    [Header("控制器参数")]
    public Parameter para;

    public double r, y; // 参考输入和当前输出 作为 ADRC 的两个输入
    public double output; // 控制器输出
    double[] state = { 0, 0, 0, 0, 0, 0 }; // 状态 x1, x2, z1, z2, z3, u_out
    // Start is called before the first frame update
    void Start()
    {
        r = 0;
        y = 0;
        output = 0;
        // ATTENTION: 控制器参数 para 在上层设置
    }

    // Update is called once per frame
    void Update()
    {

    }

    private void FixedUpdate()
    {
        double h = Time.fixedDeltaTime; // step

        // TD 对 r 进行过渡过程配置
        double x1 = state[0] + h * state[1];
        double fh = fhan(state[0] - r, state[1], para.r0, para.h0);
        double x2 = state[1] + h * fh;

        // NLSEF
        // u0 = alpha1 * fal(e1, a1, h) + alpha2 * fal(e2, a2, h);
        double e1 = state[0] - state[2]; // e1 = x1 - z1
        double e2 = state[1] - state[3]; // e2 = x2 - z2
        double u0 = para.beta1 * fal(e1, para.a1, h) + para.beta2 * fal(e2, para.a2, h);
        double u_out = (u0 + state[4]) / para.b0; // u = (u0 + z3) / b0

        // ESO y = u(2) u = x[5]
        double e = state[2] - y;
        double z1 = state[2] + h * state[3] - para.beta01 * e;
        double z2 = state[3] + h * (state[4] + para.b0 * state[5]) - para.beta02 * fal(e, 0.5, h);
        double z3 = state[4] - para.beta03 * fal(e, 0.25, h);

        state = new double[] { x1, x2, z1, z2, z3, u_out }; // 更新状态
        output = u_out; // 更新输出
    }

    double fhan(double x1, double x2, double r0, double h0) // fhan function for adrc
    {
        double d = r0 * Math.Pow(h0, 2);
        double a0 = h0 * x2;
        double y = x1 + a0;
        double a1 = Math.Sqrt(d * (d + 8 * Math.Abs(y)));
        double a2 = a0 + Math.Sign(y) * (a1 - d) / 2;
        double sy = (Math.Sign(y + d) - Math.Sign(y - d)) / 2;
        double a = (a0 + y - a2) * sy + a2;
        double sa = (Math.Sign(a + d) - Math.Sign(a - d)) / 2;
        double fh = -r0 * (a / d - Math.Sign(a)) * sa - r0 * Math.Sign(a);
        return fh;
    }

    double fal(double x, double a, double delta) // fal function for adrc
    {
        double fa;
        if (Math.Abs(x) <= delta)
        {
            fa = x / Math.Pow(delta, 1 - a);
        }
        else
        {
            fa = Math.Sign(x) * Math.Pow(Math.Abs(x), a);
        }
        return fa;
    }

}
