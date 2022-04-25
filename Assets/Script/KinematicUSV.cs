using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System;

public class KinematicUSV : MonoBehaviour
{
    public double tau_u, tau_r; // 控制信号

    public double u, v, r, x, y, psi;// 状态

    public float MAX_F = 50; // 推进器最大推力限定


    // Start is called before the first frame update
    void Start()
    {
        // TODO 设置初始值
        // TODO 如果跟踪效果不佳, 放宽 F 限制条件
        Debug.Log("start!");
    }

    // Update is called once per frame
    void Update()
    {
        transform.position = new Vector3((float)y, transform.position[1], (float)x);
        transform.rotation = Quaternion.Euler(0, (float)(psi / Math.PI * 180), 0);
    }

    private void FixedUpdate()
    {
        // 控制信号转换为推进器的力, 进行约束
        double F1 = (tau_u + tau_r) / 2.0;
        double F2 = (tau_u - tau_r) / 2.0;
        F1 = F1 > MAX_F ? MAX_F : (F1 < 0 ? 0 : F1);
        F2 = F2 > MAX_F ? MAX_F : (F2 < 0 ? 0 : F2);
        double tau1 = F1 + F2;
        double tau2 = F1 - F2;

        double du = 0.1 * r * r + v * r - 5.0 * u / 33.0 + tau1 / 33.0;
        double dv = 50.0 / 967.0 * r - 5000.0 / 31911.0 * v - r * u - 10.0 / 967.0 * tau2;
        double dr = -500.0 / 967.0 * r + 50.0 / 967.0 * v + 100.0 / 967.0 * tau2;
        double dx = Math.Cos(psi) * u - Math.Sin(psi) * v;
        double dy = Math.Sin(psi) * u + Math.Cos(psi) * v;
        double dpsi = r; // psi rad/s

        u = u + Time.fixedDeltaTime * du;
        v = v + Time.fixedDeltaTime * dv;
        r = r + Time.fixedDeltaTime * dr;
        x = x + Time.fixedDeltaTime * dx;
        y = y + Time.fixedDeltaTime * dy;
        psi = psi + Time.fixedDeltaTime * dpsi;
    }
}
