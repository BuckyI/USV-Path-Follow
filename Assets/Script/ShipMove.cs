using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ShipMove : MonoBehaviour
{
    private Vector3 speed;

    // Start is called before the first frame update
    void Start()
    {
        speed = new Vector3(1, 0, 1);
    }

    // Update is called once per frame
    void Update()
    {
        // transform.position = transform.position + new Vector3(1 * 1 * Time.deltaTime, 0, 0);
        SpeedForward();
    }

    void SpeedForward()
    {
        // transform.position = transform.position + speed * Time.deltaTime;
        Vector3 update = speed * Time.deltaTime;
        transform.LookAt(transform.position + update);
        transform.Translate(update, Space.World);
        Debug.Log(transform.position);
    }
}
