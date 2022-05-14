using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Ceto;

using System.IO;
using System.Text;

public class Disturbance : MonoBehaviour
{
    private Transform obj;
    private KinematicUSV model;

    [Header("记录扰动与船速至 csv (测试)")]
    public bool record;
    public string path;
    private StreamWriter recorder;
    private Vector2 total_disturb = new Vector2(0, 0);
    [Range(0, 1)]
    [Tooltip("一个系数用于调整扰动程度")]
    public float degree = 0.5f; // 一个系数用于调整扰动程度
    // Start is called before the first frame update
    void Start()
    {
        degree = 0.0f;
        obj = GetComponent<Transform>();
        model = GetComponent<KinematicUSV>();

        // 记录扰动大小
        record = false;
        path = Application.dataPath + "/data.csv";
        recorder = new StreamWriter(path, true, Encoding.UTF8);
    }

    // Update is called once per frame
    void FixedUpdate()
    {
        // 添加一个随波浪流动的扰动效果
        DriftInCurrent2(Time.deltaTime);
    }

    void DriftInCurrent(float deltaTime) // 随波浪流
    {
        WaveQuery query = new WaveQuery(transform.position);
        Ocean.Instance.QueryWaves(query);
        float dx = query.result.displacementX;
        float dz = query.result.displacementZ;
        float speed = Ocean.Instance.Spectrum.waveSpeed;
        // 默认波速 1m/s, 另外由水的密度, 船的质量影响, 扰动带来的位移不会同速, 这里取 0.5
        Vector3 velocity = new Vector3(dx, 0.0f, dz) * speed * deltaTime * degree;
        // 通过修改船的位置实现扰动效果, 但后来发现这样控制器观测不到扰动.
        transform.position = transform.position + velocity;
    }

    void DriftInCurrent2(float deltaTime) // 随波浪流
    {
        WaveQuery query = new WaveQuery(transform.position);
        Ocean.Instance.QueryWaves(query);
        float dx = query.result.displacementX;
        float dz = query.result.displacementZ;
        float speed = Ocean.Instance.Spectrum.waveSpeed;

        // 以波速为基础设置附加扰动
        Vector2 disturb = new Vector2(dx, dz) * speed * degree / 100.0f;
        float psi = (float)model.psi;
        float u = (float)model.u;
        Vector2 origial = new Vector2(u * Mathf.Sin(psi), u * Mathf.Cos(psi));
        Vector2 result = origial + disturb; // 原始速度与扰动速度叠加

        // 把当前时间, 船速, 累积的扰动速度大小(反映了扰动的变化幅度)
        total_disturb += disturb;
        if (record)
        {
            string info = Time.timeAsDouble + "," + u + "," + total_disturb.magnitude;
            recorder.WriteLine(info);
        }

        // 这里需要修改 kinematic 模型的速度和角度, 才能让控制器观察到扰动
        model.u = result.magnitude;
        model.psi = -Vector2.SignedAngle(Vector2.up, result) * Mathf.Deg2Rad;
    }
    void OnApplicationQuit() // 程序退出时关闭 recoreder
    {
        recorder.Close();
    }
}