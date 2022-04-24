using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System;
using System.Text.RegularExpressions;
public class LoadMovement : MonoBehaviour
{
    public TextAsset pathT;
    // 每组数据之间相距 0.1 s
    static public List<float[]> path = new List<float[]>();

    // Start is called before the first frame update
    void Start()
    {
        // load way points from path.csv
        // TextAsset pathT = Resources.Load<TextAsset>("PathData/final_path");
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
        GetComponent<Transform>().position = new Vector3(init[0], 0, init[1]);
    }

    // Update is called once per frame
    void Update()
    {

    }
}
