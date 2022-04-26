using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System;
using System.Text.RegularExpressions;

public class Task // 任务包括船, 路径点序列
{
    public GameObject ship;
    public List<Vector2> path;
    public int position = 0; // index of path
    public MoveController controller;
    public bool finished = false;
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

            GameObject ship = GeneUSV();

            // 调整船的初始位置于路径点起点
            // ATTENTION: 要修改 Kinematic USV 的变量 而非 transform
            // 路径点坐标系和船的坐标系不一样
            Vector2 pos = task.path[task.position];
            ship.GetComponent<KinematicUSV>().x = pos.y;
            ship.GetComponent<KinematicUSV>().y = pos.x;

            // Add Controller
            MoveController obj = (MoveController)ship.AddComponent(typeof(MoveController));

            task.controller = obj;
            task.ship = ship;
            tasks.Add(task);
        }
    }

    // Update is called once per frame
    void Update()
    {

    }

    private void FixedUpdate()
    {
        foreach (Task item in tasks)
        {
            float[] refSignal = calc_ref(item);
            item.controller.ref_u = refSignal[0];
            item.controller.ref_yaw = refSignal[1];

            for (int j = item.position; j < item.path.Count - 1; j++)
            {
                Vector3 p1 = new Vector3(item.path[j].x, 10, item.path[j].y);
                Vector3 p2 = new Vector3(item.path[j + 1].x, 10, item.path[j + 1].y);
                Debug.DrawLine(p1, p2, Color.white, 0);
            }
            // Debug.Log(item.position.ToString());
        }
    }
    float[] calc_ref(Task t)
    {
        KinematicUSV data = t.ship.GetComponent<KinematicUSV>();
        Vector2 location = data.GetLocation(); // (x,y)
        float psi = (float)data.psi; // rad
        List<Vector2> path = t.path;

        // u: 期望速度
        // yaw: 期望船角度
        float u = 0.1f;
        float yaw = 0;

        // 确定目标点 the closest point which has't been accessed 
        Vector2 target = path[t.position];
        Vector2 d = target - location;   // 期望更新矢量
        while ((d.sqrMagnitude < 100) && (t.position < path.Count - 1))
        {
            t.position = t.position + 1; // 5m 之内视为已观测, 移到下一个点
            target = path[t.position];
            d = target - location;
        }

        // 计算给定值
        if (t.finished || ((t.position == (path.Count - 1)) && d.sqrMagnitude < 4))
        {
            // 目标点是终点, 且船已经到达终点 2m 范围内, 视为结束
            t.finished = true;
            u = 0;
            yaw = psi;
            return new float[] { u, yaw };
        }
        else
        {
            // 计算期望角度值 rad
            yaw = -Vector2.SignedAngle(Vector2.up, d) * Mathf.Deg2Rad; // 加负号是因为 AngleY 是顺时针
            while (Mathf.Abs(yaw - psi) > Mathf.PI)
            {
                if (yaw - psi > Mathf.PI) yaw = yaw - 2 * Mathf.PI;
                else if (yaw - psi < -Mathf.PI) yaw = yaw + 2 * Mathf.PI;
            }

            if (Mathf.Abs(yaw - psi) > 0.5 * Mathf.PI)
            {
                // 如果发现目标点在船的后面(有跟踪失败的风险)
                // 则策略切换到选择最近点作为目标点
                int index = t.position;
                float min_distance = d.sqrMagnitude;
                for (int i = t.position + 1; i < t.path.Count; i++)
                {
                    float distance = (path[i] - location).sqrMagnitude;
                    if (distance < min_distance)
                    {
                        min_distance = distance;
                        index = i;
                    }
                }
                t.position = index + 5; // 跟踪点转移到最近点, 下次更新生效
                // +5 往前看, 不加也行
            }

            // 计算期望速度值 (考虑距离和转角)
            // u = 2 * Mathf.Log10(d.magnitude + 1) * (1 + Mathf.Cos(yaw - psi)) * 0.5f;
            u = 2 * Mathf.Log10(d.magnitude + 1);
            return new float[] { u, yaw };
        }

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
                // waypoints located at (x, z) in Unity
                single_path.Add(new Vector2(x, y)); // 这里坐标进行转换
            }
        }
        Debug.Log(single_path.Count + " waypoints loaded.");
        return single_path;
    }

}


/*
在 Unity 里面 psi 的位置好像有点不一样
location x, y
当前 psi
path nx2 数组


function [u, yaw, reached] = calc_ref(location,psi, path)
% u: 期望速度
% yaw: 期望船角度
% reached: 是否到达终点
u=0.1;
yaw=0;
reached=false;

%% 确定目标点
% the closest point which has't been accessed 
persistent k;
if isempty(k)
    k=1;
end

while k<length(path)
    target=path(k,:);
    d=target-location'; % 期望更新矢量
    if sum(d.^2)<25 && k<length(path) % 5m 之内视为已观测, 移到下一个点
       k=k+1;
    else
        break;
    end
end

d=path(k,:)-location'; % 期望更新矢量
if sum(d.^2)<4 % 船距离终点2米, 视为到达终点, 停下来
    u=0;
    yaw=0;
    reached=true;
    return
end

%% 计算期望值
% calculate yaw
yaw=calc_yaw(d);
while abs(yaw-psi)>pi
    if yaw-psi>pi
        yaw=yaw-2*pi;
    elseif yaw-psi<-pi
        yaw=yaw+2*pi;
    end
end

% 如果发现目标点在船的后面(有跟踪失败的风险),则策略切换到选择最近点作为目标点
if abs(yaw-psi)>0.5*pi
    subpath=path(k:end,:);
    dir=subpath-location';
    distance=sum(dir.^2,2); % x^2+y^2
    min_index = find(distance==min(distance));
    k=k + min_index(1)-1; % 找到第一个距离最近的索引, 并转化成相对path的索引
    % 切换 k, 下次更新
end

% calculate u
u=log10(sqrt(sum(d.^2,"all"))+1)*(1+cos(yaw-psi))*0.5; % 考虑到距离, 转角确定速度

function yaw=calc_yaw(d)
    yaw=0;
    if d(1)>0
        yaw=atan(d(2)/d(1));
    elseif d(1)<=0 && d(2) >0
        yaw=pi+atan(d(2)/d(1));
    elseif d(1)<=0 && d(2)<=0
        yaw=-pi+atan(d(2)/d(1));
    end


*/
