using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class MoveControl : MonoBehaviour
{
    private float movementSpeed = 10f;
    public float sensitivityMouse = 2f;
    public float sensitivetyKeyBoard = 0.1f;
    public float sensitivetyMouseWheel = 10f;

    public Vector3 offset;

    float m_Height = 100f; //相机距离人物高度

    float m_Distance = 100f; //相机距离人物距离

    float m_Speed = 4f; //相机跟随速度

    Vector3 m_TargetPosition; //目标位置

    public List<Transform> follows; //要跟随的东西
    int fi = 0; // 要跟随的序列

    // Start is called before the first frame update
    void Start()
    {
    }

    // Update is called once per frame
    void Update()
    {
        if (Input.GetKey("enter")) // 切换跟踪对象
        {
            fi++;
            if (fi >= follows.Count) fi -= follows.Count;
        }
        Transform follow = follows[fi];
        Follow(follow);
    }
    void Follow(Transform follow)
    {
        // x, z 跟随 usv
        transform.position = offset + follow.position - new Vector3(0, follow.position[1], 0);
        // transform.position = offset + ship.position;
    }
    void ManualAdjust()
    {
        if (Input.GetAxis("Mouse ScrollWheel") != 0)
        {
            this.GetComponent<Camera>().fieldOfView = this.GetComponent<Camera>().fieldOfView - Input.GetAxis("Mouse ScrollWheel") * sensitivetyMouseWheel;
        }
        //get the Input from Horizontal axis
        float horizontalInput = Input.GetAxis("Horizontal");
        //get the Input from Vertical axis
        float verticalInput = Input.GetAxis("Vertical");

        //update the position
        transform.position = transform.position + new Vector3(horizontalInput * movementSpeed * Time.deltaTime, 0, verticalInput * movementSpeed * Time.deltaTime);
    }

    void SmoothFollow(Transform follow)
    {
        //得到这个目标位置
        m_TargetPosition = follow.position + Vector3.up * m_Height - follow.forward * m_Distance;
        //相机位置
        transform.position = Vector3.Lerp(transform.position, m_TargetPosition, m_Speed * Time.deltaTime);
        //相机时刻看着人物
        transform.LookAt(follow);
    }
}
