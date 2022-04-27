using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Ceto;

public class Disturbance : MonoBehaviour
{
    private Transform obj;
    // Start is called before the first frame update
    void Start()
    {
        obj = GetComponent<Transform>();

    }

    // Update is called once per frame
    void Update()
    {
        // 默认的 BoyantStructure 计算浮力并进行更新, 只会影响船的高度以及 rotation
        // 其中会对船的 psi 进行干扰(幅度比较小, 只有 0.x°)
        // 所以添加一个随波浪流动的干扰效果, 还是比较显著的
        DriftInCurrent(Time.deltaTime);
    }

    void DriftInCurrent(float deltaTime) // 随波浪流
    {
        WaveQuery query = new WaveQuery(transform.position);
        Ocean.Instance.QueryWaves(query);
        float dx = query.result.displacementX;
        float dz = query.result.displacementZ;
        float speed = Ocean.Instance.Spectrum.waveSpeed; // 默认波速 1m/s
        Vector3 velocity = new Vector3(dx, 0.0f, dz) * speed * deltaTime;
        transform.position = transform.position + velocity;
    }
}
