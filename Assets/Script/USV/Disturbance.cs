using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Ceto;

public class Disturbance : MonoBehaviour
{
    private Transform obj;
    [Range(0, 1)]
    public float degree = 0.5f; // 一个系数用于调整扰动程度
    // Start is called before the first frame update
    void Start()
    {
        obj = GetComponent<Transform>();
    }

    // Update is called once per frame
    void FixedUpdate()
    {
        // 添加一个随波浪流动的扰动效果
        DriftInCurrent(Time.deltaTime);
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
        transform.position = transform.position + velocity;
    }
}
