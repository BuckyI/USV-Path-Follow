using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System;
using System.Text.RegularExpressions;
public class LoadMovement : MonoBehaviour
{
    static public List<float[]> path = new List<float[]>();
    // Start is called before the first frame update
    void Start()
    {
        // load way points from path.csv
        TextAsset target_ass = Resources.Load<TextAsset>("PathData/path");
        foreach (string t in target_ass.text.Split('\n'))
        {

            Match m = Regex.Match(t, @"(\d+),\s?(\d+),\s?(-?\d+)");
            if (m.Success)
            {
                float x = (float)Convert.ToDouble(m.Groups[1].Value);
                float y = (float)Convert.ToDouble(m.Groups[2].Value);
                float psi = (float)Convert.ToDouble(m.Groups[3].Value);
                path.Add(new float[] { x, y, psi }); // store
            }
        }
        Debug.Log(path.Count + "waypoints loaded.");
    }

    // Update is called once per frame
    void Update()
    {

    }
}
