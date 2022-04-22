using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System;
using System.Text.RegularExpressions;

public class TestScript : MonoBehaviour
{
    private List<float[]> targets = new List<float[]>();
    // Start is called before the first frame update
    void Start()
    {
        TextAsset target_ass = Resources.Load<TextAsset>("PathData/targets");
        foreach (string t in target_ass.text.Split('\n'))
        {

            Match m = Regex.Match(t, @"(\d+),\s?(\d+)");
            if (m.Success)
            {
                float x = (float)Convert.ToDouble(m.Groups[1].Value);
                float y = (float)Convert.ToDouble(m.Groups[2].Value);
                targets.Add(new float[] { x, y });
            }
        }
        Debug.Log(targets.Count);

        // Debug.Log(targets.text.Split('\n'));
    }

    // Update is called once per frame
    void Update()
    {

    }
}
