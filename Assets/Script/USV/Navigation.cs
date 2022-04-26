using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System;
using System.Text.RegularExpressions;

public struct Task // 任务包括船, 路径点序列
{
    public GameObject ship;
    public List<Vector2> path;
    public int position; // index of path
}

public class Navigation : MonoBehaviour
{
    public TextAsset[] pathsAssets; // 输入数据
    public List<Task> tasks = new List<Task>();
    // Start is called before the first frame update
    void Start()
    {
        transform.position = Vector3.zero;
        foreach (TextAsset item in pathsAssets)
        {
            Task task = new Task();
            task.path = ReadPath(item);
            task.position = 0;
            task.ship = GeneUSV();

            // 调整船的初始位置于路径点起点
            // ATTENTION: 要修改 Kinematic USV 的变量 而非 transform
            // 路径点坐标系和船的坐标系不一样
            Vector2 pos = task.path[task.position];
            task.ship.GetComponent<KinematicUSV>().x = pos.y;
            task.ship.GetComponent<KinematicUSV>().y = pos.x;

            tasks.Add(task);
        }
    }

    // Update is called once per frame
    void Update()
    {

    }

    GameObject GeneUSV()
    {
        GameObject go = GameObject.Instantiate(Resources.Load("Prefabs/USV Variant")) as GameObject;
        go.transform.SetParent(GetComponent<Transform>());
        // 摄像机跟随此目标
        GameObject.Find("Main Camera").GetComponent<MoveControl>().follows.Add(go.transform);
        return go;
    }

    List<Vector2> ReadPath(TextAsset item) // read waypoints from csv
    {
        List<Vector2> single_path = new List<Vector2>();
        foreach (string t in item.text.Split('\n'))
        {
            Match m = Regex.Match(t, @"([\d\.]+),\s?([\d\.]+)");
            if (m.Success)
            {
                float x = (float)Convert.ToDouble(m.Groups[1].Value);
                float y = (float)Convert.ToDouble(m.Groups[2].Value);
                single_path.Add(new Vector2(x, y)); // store
            }
        }
        Debug.Log(single_path.Count + " waypoints loaded.");
        return single_path;
    }
