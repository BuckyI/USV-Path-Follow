using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System;

public class ADRC : MonoBehaviour
{
    public double r, y; // 参考输入和当前输出 作为 ADRC 的两个输入
    public double[] state; // 状态 x1, x2, z1, z2, z3, u_out
    // Start is called before the first frame update
    void Start()
    {

    }

    // Update is called once per frame
    void Update()
    {

    }

    private void FixedUpdate()
    {

    }

    double fhan(double x1, double x2, double r0, double h0) // % fhan function for adrc
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

    double fal(double x, double a, double delta) // % fal function for adrc
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
