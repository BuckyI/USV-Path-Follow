using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System;

public class ShipMove : MonoBehaviour
{
    private Vector3 speed = new Vector3(1, 0, 1);
    private float total_time = 0;
    public float accelerate = 1; // 用于加速动画
    // Start is called before the first frame update
    void Start()
    {

    }

    // Update is called once per frame
    void Update()
    {
        total_time += Time.deltaTime;
        // transform.position = transform.position + new Vector3(1 * 1 * Time.deltaTime, 0, 0);
        // SpeedForward();
        FollowPath();
    }

    void SpeedForward()
    {
        // transform.position = transform.position + speed * Time.deltaTime;
        Vector3 update = speed * Time.deltaTime;
        transform.LookAt(transform.position + update);
        transform.Translate(update, Space.World);
        Debug.Log(transform.position);
    }

    void FollowPath()
    {
        int count = (int)(total_time * accelerate / 0.1);
        // stop when get destination
        if (count >= LoadMovement.path.Count) count = LoadMovement.path.Count - 1;

        float[] info = LoadMovement.path[count]; // x, y, psi
        float x = info[0];
        float y = info[1];
        float psi = info[2];

        // transform.position = new Vector3(info[0], transform.position[1], info[1]);
        transform.position = new Vector3(x, 0, y);

        // float diff=
        // Vector3 angle = new Vector3((float)Math.Sin(Math.PI * psi / 180), 0, (float)Math.Cos(Math.PI * psi / 180));
        // transform.rotation = Quaternion.FromToRotation(Vector3.forward, angle);
        transform.eulerAngles = new Vector3(0, psi, 0);
    }
}
