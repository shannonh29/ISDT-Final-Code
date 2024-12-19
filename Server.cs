using System.Collections;
using System.Collections.Generic;
using System;
using System.Text;
using System.Net;
using System.Net.Sockets;
using System.Threading;
using UnityEngine;

// todo:
// add more anchors and place them in the whole area where we will operate
// place camera further away
// take average of 2-3 frames if mug is shaking
// change hirachy: move moving components to extra space so it doesn't get affected by their position

public class Server : MonoBehaviour
{
    const string hostIP = "192.168.1.146"; // Select your IP
    const int port = 80; // Select your port
    TcpListener server = null;
    TcpClient client = null;
    NetworkStream stream = null;
    Thread thread;

    // Define your own message
    [Serializable]
    public class Message
    {
        //public string some_string;
        public int id;
        public float x;
        public float y;
        public float z;
    }

    // [Serializable]
    // public class Aruco {
    //     public float x;
    //     public float y;
    //     public float z;
    // }

    private int anchor_amt = 5;//CHANGE THIS TO HOW MANY ANCHORS YOU WANT

    private float timer = 0;
    private static object Lock = new object();  // lock to prevent conflict in main thread and server thread
    private List<Message> MessageQue = new List<Message>();

    private bool callibration = false; //used to see if callibration is done

    private int old_length = 0;

    private void Start()
    {
        thread = new Thread(new ThreadStart(SetupServer));
        thread.Start();
    }

    private void Update()
    {
        // Send message to client every 2 second
        if(Time.time > timer)
        {
            OVRSpatialAnchor[] anchors = FindObjectsOfType<OVRSpatialAnchor>();

            //Vector3 position = transform.position;
            Debug.Log("Number of OVRSpatialAnchor objects in the scene: " + anchors.Length);

            if(anchors.Length > old_length && callibration == false) {
                Debug.Log("Please Callibrate");

                // Get the position of the latest anchor
                Vector3 position = anchors[anchors.Length - 1].transform.position;
            
                Message msg = new Message();
                msg.id = anchors.Length;
                msg.x = position.x;
                msg.y = position.y;
                msg.z = position.z;

                // Log the message for debugging
                Debug.Log($"New Anchor Position: X = {msg.x}, Y = {msg.y}, Z = {msg.z}");

                SendMessageToClient(msg);

                if(anchors.Length == anchor_amt) {
                    callibration = true;
                } else {
                    old_length = anchors.Length;
                }
            }

            //Debug.Log($"Position: X = {msg.x}, Y = {msg.y}, Z = {msg.z}");

            //SendMessageToClient(msg);
            timer = Time.time + 2f;
        }

        // Process message que

        //Debug.Log("Abot to be locked");
        lock(Lock)
        {
            //Debug.Log($"MessageQue Count: {MessageQue.Count}");
            foreach (Message message in MessageQue)
            {
                // Unity only allow main thread to modify GameObjects.
                // Spawn, Move, Rotate GameObjects here. 
                Debug.Log("Received");
                Vector3 newPosition = new Vector3(message.x, message.y, message.z);
                transform.position = newPosition; // Update position
                //Debug.Log("Received Str: " + message.some_string + " Int: " + message.some_int + " Float: " + message.some_float);
            }
            MessageQue.Clear();
        }

        //Debug.Log("Finished updating");
    }

    private void SetupServer()
    {
        try
        {
            IPAddress localAddr = IPAddress.Parse(hostIP);
            server = new TcpListener(localAddr, port);
            server.Start();

            byte[] buffer = new byte[1024];
            string data = null;

            while (true)
            {
                Debug.Log("Waiting for connection...");
                client = server.AcceptTcpClient();
                Debug.Log("Connected!");

                data = null;
                stream = client.GetStream();

                // Receive message from client    
                int i;
                while ((i = stream.Read(buffer, 0, buffer.Length)) != 0)
                {
                    data = Encoding.UTF8.GetString(buffer, 0, i);
                    Message message = Decode(data);
                    // Add received message to que
                    lock(Lock)
                    {
                        MessageQue.Add(message);
                    }
                }
                client.Close();
            }
        }
        catch (SocketException e)
        {
            Debug.Log("SocketException: " + e);
        }
        finally
        {
            server.Stop();
        }
    }

    private void OnApplicationQuit()
    {
        stream.Close();
        client.Close();
        server.Stop();
        thread.Abort();
    }

    public void SendMessageToClient(Message message)
    {
        if(stream != null && stream.CanWrite) {
            byte[] msg = Encoding.UTF8.GetBytes(Encode(message));
            stream.Write(msg, 0, msg.Length);
            Debug.Log("Sent: " + message);
        } else {
            Debug.Log("Stream is null or not writable at this time");
        }
    }

    // Encode message from struct to Json String
    public string Encode(Message message)
    {
        return JsonUtility.ToJson(message, true);
    }

    // Decode messaage from Json String to struct
    public Message Decode(string json_string)
    {
        Message msg = JsonUtility.FromJson<Message>(json_string);
        return msg;
    }
}

