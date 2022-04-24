using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class LoadUSVs : MonoBehaviour
{
    public TextAsset[] paths; // Inspector 分配路径
    // Start is called before the first frame update
    void Start()
    {
        // transform.position = new Vector3(0, 0, 0); // 初始位置设在原点
        foreach (TextAsset item in paths)
        {
            GeneUSV(item);
            Debug.Log("generate one usv");
        }
    }

    // Update is called once per frame
    void Update()
    {

    }

    GameObject GeneUSV(TextAsset path)
    {
        GameObject go = GameObject.Instantiate(Resources.Load("Prefabs/USV Variant")) as GameObject;
        go.transform.SetParent(GetComponent<Transform>());
        go.GetComponent<ShipMove>().pathT = path; // 给每艘船配置不同的路径
        GameObject.Find("Main Camera").GetComponent<MoveControl>().follows.Add(go.transform); // 摄像机跟随目标
        return go;
    }
}
