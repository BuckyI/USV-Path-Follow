using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System;
using System.Text.RegularExpressions;
public class LoadTargets : MonoBehaviour
{
    private List<float[]> targets = new List<float[]>();
    // Start is called before the first frame update
    void Start()
    {
        // load target location from targets.csv
        TextAsset target_ass = Resources.Load<TextAsset>("PathData/targets");
        foreach (string t in target_ass.text.Split('\n'))
        {

            Match m = Regex.Match(t, @"(\d+),\s?(\d+)");
            if (m.Success)
            {
                float x = (float)Convert.ToDouble(m.Groups[1].Value);
                float y = (float)Convert.ToDouble(m.Groups[2].Value);
                targets.Add(new float[] { x, y }); // store
            }
        }
        Debug.Log(targets.Count + "targets loaded.");

        // generate prefabs
        transform.position = new Vector3(0, 0, 0); // 初始位置设在原点
        foreach (float[] item in targets)
        {
            GameObject go = GameObject.Instantiate(Resources.Load("Prefabs/Target")) as GameObject;
            go.transform.position = new Vector3(item[0], 0, item[1]);
            go.transform.SetParent(GetComponent<Transform>());
        }

    }

    // Update is called once per frame
    void Update()
    {

    }
}
