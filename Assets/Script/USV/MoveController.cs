using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class MoveController : MonoBehaviour
{
    [Header("控制器")]
    public ADRC controller_u;
    public ADRC controller_yaw;
    [Header("控制器参数")]
    public Parameter para_u;
    public Parameter para_yaw;
    [Header("给定值")]
    public double ref_u;
    public double ref_yaw;
    private KinematicUSV model;
    // Start is called before the first frame update
    void Start()
    {
        #region 速度控制器参数 para_u
        // Ts
        para_u.h = Time.fixedDeltaTime; // 0.01

        // TD para_u
        para_u.r0 = 0.1;
        para_u.h0 = 0.01;
        // ESO para_u
        para_u.beta01 = 1;
        para_u.beta02 = 3;
        para_u.beta03 = 10;
        para_u.b0 = 0.1;
        // NLSEF para_u
        para_u.beta1 = 100;
        para_u.beta2 = 4.5;
        para_u.a1 = 0.75;
        para_u.a2 = 1.5;
        #endregion

        #region 航向控制器参数 para_yaw
        // Ts
        para_yaw.h = Time.fixedDeltaTime;
        // TD para_yaw
        para_yaw.r0 = 2;
        para_yaw.h0 = 0.01;
        // ESO para_yaw
        para_yaw.beta01 = 1;
        para_yaw.beta02 = 2;
        para_yaw.beta03 = 10;
        para_yaw.b0 = 0.02;
        // NLSEF para_yaw
        para_yaw.beta1 = 10;
        para_yaw.beta2 = 13;
        para_yaw.a1 = 0.75;
        para_yaw.a2 = 1.5;
        #endregion

        controller_u = (ADRC)gameObject.AddComponent(typeof(ADRC));
        controller_yaw = (ADRC)gameObject.AddComponent(typeof(ADRC));
        controller_u.para = para_u;
        controller_yaw.para = para_yaw;

        ref_u = 1;
        ref_yaw = 1.7;

        model = GetComponent<KinematicUSV>(); // 用于获取船的状态信息
    }

    // Update is called once per frame
    void Update()
    {

    }

    private void FixedUpdate()
    {
        // input ref_u, ref_w
        controller_u.r = ref_u;
        controller_u.y = model.u;
        controller_yaw.r = ref_yaw;
        controller_yaw.y = model.psi;

        // update

        // output
        model.tau_u = controller_u.output;
        model.tau_r = controller_yaw.output;
    }
}
