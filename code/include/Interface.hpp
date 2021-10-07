
#ifndef INTERFACE_HPP
#define INTERFACE_HPP

#include <vector>
#include <thread>
#include <memory>
#include <functional>
#include <zmq.hpp>
// #include <zhelpers.hpp>
#include <iostream>
#include <msgpack.hpp>
#include "Simulation_IO.h"
// template <typename T>

class Client {
public:
    Client()
        : ctx_(1),
        client_socket_(ctx_, ZMQ_REQ)
    {}

    void start() {
        
        std::cout << "Connecting to hello world server…" << std::endl;
        client_socket_.connect("tcp://localhost:5555");

    }

    
    void send_okay(){
        
        // msgpack::sbuffer sbuf;
        // msgpack::pack(sbuf, ("okay"));

        // zmq::message_t msg(sbuf.size());
   
        // memcpy(msg.data(), sbuf.data(), sbuf.size());

        // client_socket_.send(msg);

        std::string msgToClient("greeting from C++");
        zmq::message_t reply(msgToClient.size());
        memcpy((void *) reply.data(), (msgToClient.c_str()), msgToClient.size());
        client_socket_.send(reply);

    }
    
    void send_real_time_data(std::vector<float> data_vector){

        msgpack::sbuffer sbuf;
        msgpack::pack(sbuf, data_vector[0]);
        msgpack::pack(sbuf, data_vector[1]);
        msgpack::pack(sbuf, data_vector[2]);
      

        zmq::message_t msg(sbuf.size());
        memcpy((void *) msg.data(), sbuf.data(), sbuf.size());
        client_socket_.send(msg);

        std::cout << "Sending data...: " << std::endl;


    }

    void send_data(output* data){
        
        // std::cout << "Sending data...: "<< data << std::endl;

        // create a msgpack to send to the communicator 
        msgpack::sbuffer sbuf;

        // the length of each data is equal to the number of whiskers
        msgpack::pack(sbuf, data->Fx);
        msgpack::pack(sbuf, data->Fy);
        msgpack::pack(sbuf, data->Fz);
        msgpack::pack(sbuf, data->Mx);
        msgpack::pack(sbuf, data->My);
        msgpack::pack(sbuf, data->Mz);

        // create matrix (2D vectors) to store data of each whisker
        std::vector<std::vector<int>> C_mat;
        std::vector<std::vector<float>> X_mat;
        std::vector<std::vector<float>> Y_mat;
        std::vector<std::vector<float>> Z_mat;

        // access each whisker (whisker loop)
        for (int whi=0; whi < data->Q.size();whi++){
            std::vector<int> c_push; //vector to push into C_mat row
            std::vector<float> x_push;
            std::vector<float> y_push;
            std::vector<float> z_push;

            // coloum loop since whisker is divided into 20 segments
            for (int col=0; col < data->Q[whi].C[0].size(); col++){
                c_push.push_back(data->Q[whi].C[0][col]);
                x_push.push_back(data->Q[whi].X[0][col]);
                y_push.push_back(data->Q[whi].Y[0][col]);
                z_push.push_back(data->Q[whi].Z[0][col]); 
            }
            // push vectors into each row
            C_mat.push_back(c_push);
            X_mat.push_back(x_push);
            Y_mat.push_back(y_push);
            Z_mat.push_back(z_push);
        }
        msgpack::pack(sbuf, C_mat);
        msgpack::pack(sbuf, X_mat);
        msgpack::pack(sbuf, Y_mat);
        msgpack::pack(sbuf, Z_mat);
       
        zmq::message_t msg(sbuf.size());
        // std::cout << "Qsize: "<< data->Q.C.size() << "Fsize: " << data -> Fx.size() << std::endl;
        memcpy(msg.data(), sbuf.data(), sbuf.size());
        client_socket_.send(msg);

    
    }


    void receive_data(std::vector<std::vector<float>> &all_data){
        
        // std::cout << "Waiting for data..." << std::endl;
        zmq::message_t request;

        //  Wait for next request from client
        client_socket_.recv(&request);
        msgpack::sbuffer sbuf;
        sbuf.write(static_cast<const char *>(request.data()), request.size());
        
        
        // serializes multiple objects using msgpack::packer.
        msgpack::sbuffer buffer;

        // deserializes these objects using msgpack::unpacker.
        msgpack::unpacker pac;

        // feeds the buffer.
        pac.reserve_buffer(sbuf.size());
        memcpy(pac.buffer(), sbuf.data(), sbuf.size());
        pac.buffer_consumed(sbuf.size());

        

        // now starts streaming deserialization.
        msgpack::object_handle oh;
        msgpack::object obj;
        std::vector<float> rvec;
        
        while(pac.next(oh)) {
            obj = oh.get();
    
            // convert it into statically typed object
            obj.convert(rvec);
            all_data.push_back(rvec);
            
        }
        
        // std::cout << "Data received." << std::endl;
        
        // std::cout << all_data.size() << std::endl;

        // for(int i=0;i<all_data.size();i++){
        //     std::cout << all_data[i][0] << std::endl;
        // }
        
                
    }

    void pack_and_send(zmq::socket_t client, msgpack::sbuffer sbuf){
        
        zmq::message_t request(sbuf.size());
        memcpy(request.data(), sbuf.data(), sbuf.size());
        client.send(request);
    }

private:
    zmq::context_t ctx_;
    zmq::socket_t client_socket_;
};


#endif