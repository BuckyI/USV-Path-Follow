using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System;
using System.Text.RegularExpressions;
public class ShipMove : MonoBehaviour
{
    // 路径数据
    public TextAsset pathT; // 每组数据之间相距 0.1 s
    public List<float[]> path = new List<float[]>();

    // 运动数据
    private Vector3 speed = new Vector3(1, 0, 1);
    private float total_time = 0;
    public float accelerate = 1; // 用于加速动画
    // Start is called before the first frame update
    void Start()
    {
        foreach (string t in pathT.text.Split('\n'))
        {
            Match m = Regex.Match(t, @"([\d\.]+),\s?([\d\.]+),\s?(-?[\d\.]+)");
            if (m.Success)
            {
                float x = (float)Convert.ToDouble(m.Groups[1].Value);
                float y = (float)Convert.ToDouble(m.Groups[2].Value);
                float psi = (float)Convert.ToDouble(m.Groups[3].Value);
                path.Add(new float[] { x, y, psi }); // store
            }
        }
        Debug.Log(path.Count + " waypoints loaded.");

        // 初始化位置
        float[] init = path[0];
        transform.position = new Vector3(init[0], 0, init[1]);
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
        if (count >= path.Count) count = path.Count - 1;

        float[] info = path[count]; // x, y, psi
        float x = info[0];
        float y = info[1];
        float psi = info[2];

        // transform.position = new Vector3(info[0], transform.position[1], info[1]);
        transform.position = new Vector3(x, 0, y);

        // transform.rotation = Quaternion.FromToRotation(Vector3.forward, angle);
        transform.eulerAngles = new Vector3(0, psi, 0);
    }
}
