using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System;

public class KinematicUSV : MonoBehaviour
{
    public double tau_u, tau_r; // 控制信号

    public double u, v, r, x, y, psi;// 状态

    private double last_x, last_y, last_psi; // 保存之前的信息
    public float MAX_F = 50; // 推进器最大推力限定


    // Start is called before the first frame update
    void Start()
    {
        // TODO 设置初始值
        // TODO 如果跟踪效果不佳, 放宽 F 限制条件
        Debug.Log("start!");

        //保存外部对 x, y, psi 的修改
        // 船的初始信息 x, y, psi 由创建者决定
        UpdatePosition();
    }

    // Update is called once per frame
    void Update()
    {
        // transform.position = new Vector3((float)y, transform.position[1], (float)x);
        // transform.rotation = Quaternion.Euler(0, (float)(psi / Math.PI * 180), 0);
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

        // 计算外部更新, 保留外部对于模型的修改 (海浪的影响)
        double delta_x = transform.position[2] - last_x;
        double delta_y = transform.position[0] - last_y;
        double delta_psi = transform.eulerAngles[1] * Mathf.Deg2Rad - last_psi;
        // 角度的变化不会超过 PI (eulerAngles 0-360)
        while (delta_psi > Mathf.PI) delta_psi -= 2 * Mathf.PI;
        while (delta_psi < -Mathf.PI) delta_psi += 2 * Mathf.PI;
        x += delta_x;
        y += delta_y;
        psi += delta_psi;

        // 更新位置
        UpdatePosition();
    }

    public Vector2 GetLocation()
    {
        return new Vector2((float)y, (float)x); // 获得 xOz 平面下的坐标, 方便计算
    }

    private void UpdatePosition()
    {
        transform.position = new Vector3((float)y, transform.position[1], (float)x);
        Vector3 angle = transform.eulerAngles;
        angle[1] = (float)psi * Mathf.Rad2Deg;
        transform.eulerAngles = angle;

        last_x = transform.position[2];
        last_y = transform.position[0];
        last_psi = transform.eulerAngles[1] * Mathf.Deg2Rad;
    }
}
