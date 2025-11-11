import socket
import pickle
import json
import struct
import threading
from enum import Enum
from typing import Any, Optional, Callable

class SerializationMethod(Enum):
    """数据序列化方法"""
    PICKLE = "pickle"
    JSON = "json"
    STRING = "string"

class SocketServer:
    """
    Socket服务器类，支持自动数据编码解码
    """
    
    def __init__(self, 
                 host: str = 'localhost', 
                 port: int = 12345,
                 serialization_method: SerializationMethod = SerializationMethod.PICKLE,
                 max_clients: int = 5):
        """
        初始化服务器
        
        Args:
            host: 主机地址
            port: 端口号
            serialization_method: 序列化方法
            max_clients: 最大客户端连接数
        """
        self.host = host
        self.port = port
        self.serialization_method = serialization_method
        self.max_clients = max_clients
        self.server_socket = None
        self.client_sockets = []
        self.running = False
        self.message_handlers = []
        
    def start(self):
        """启动服务器"""
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(self.max_clients)
            self.running = True
            
            print(f"🚀 Server started on {self.host}:{self.port}")
            print(f"📦 Serialization method: {self.serialization_method.value}")
            
            # 启动接受客户端连接的线程
            accept_thread = threading.Thread(target=self._accept_clients, daemon=True)
            accept_thread.start()
            
        except Exception as e:
            print(f"❌ Failed to start server: {e}")
            raise
    
    def _accept_clients(self):
        """接受客户端连接"""
        while self.running:
            try:
                client_socket, client_address = self.server_socket.accept()
                print(f"✅ Client connected from {client_address}")
                
                # 为每个客户端创建独立的接收线程
                client_thread = threading.Thread(
                    target=self._handle_client,
                    args=(client_socket, client_address),
                    daemon=True
                )
                client_thread.start()
                
                self.client_sockets.append(client_socket)
                
            except Exception as e:
                if self.running:
                    print(f"❌ Error accepting client: {e}")
    
    def _handle_client(self, client_socket: socket.socket, client_address: tuple):
        """处理客户端通信"""
        try:
            while self.running:
                # 接收数据
                data = self._receive_data(client_socket)
                if data is None:
                    break
                
                # 调用消息处理器
                for handler in self.message_handlers:
                    try:
                        handler(data, client_socket, client_address)
                    except Exception as e:
                        print(f"❌ Error in message handler: {e}")
                        
        except Exception as e:
            print(f"❌ Error handling client {client_address}: {e}")
        finally:
            self._disconnect_client(client_socket, client_address)
    
    def _receive_data(self, client_socket: socket.socket) -> Any:
        """接收并解码数据"""
        try:
            # 接收数据长度前缀 (4字节)
            length_data = self._recv_exact(client_socket, 4)
            if not length_data:
                return None
            
            data_length = struct.unpack('>I', length_data)[0]
            
            # 接收实际数据
            serialized_data = self._recv_exact(client_socket, data_length)
            if not serialized_data:
                return None
            
            # 根据序列化方法解码数据
            return self._decode_data(serialized_data)
            
        except Exception as e:
            print(f"❌ Error receiving data: {e}")
            return None
    
    def _send_data(self, client_socket: socket.socket, data: Any) -> bool:
        """编码并发送数据"""
        try:
            serialized_data = self._encode_data(data)
            data_length = len(serialized_data)
            
            # 发送数据长度前缀
            length_prefix = struct.pack('>I', data_length)
            client_socket.sendall(length_prefix + serialized_data)
            return True
            
        except Exception as e:
            print(f"❌ Error sending data: {e}")
            return False
    
    def _encode_data(self, data: Any) -> bytes:
        """编码数据为字节"""
        if self.serialization_method == SerializationMethod.PICKLE:
            return pickle.dumps(data)
        elif self.serialization_method == SerializationMethod.JSON:
            if isinstance(data, (str, int, float, bool, list, dict, type(None))):
                return json.dumps(data).encode('utf-8')
            else:
                # 对于不支持JSON序列化的对象，使用pickle
                return pickle.dumps(data)
        elif self.serialization_method == SerializationMethod.STRING:
            if isinstance(data, str):
                return data.encode('utf-8')
            else:
                return str(data).encode('utf-8')
        else:
            return pickle.dumps(data)
    
    def _decode_data(self, serialized_data: bytes) -> Any:
        """从字节解码数据"""
        try:
            if self.serialization_method == SerializationMethod.PICKLE:
                return pickle.loads(serialized_data)
            elif self.serialization_method == SerializationMethod.JSON:
                return json.loads(serialized_data.decode('utf-8'))
            elif self.serialization_method == SerializationMethod.STRING:
                return serialized_data.decode('utf-8')
            else:
                return pickle.loads(serialized_data)
        except Exception as e:
            print(f"❌ Error decoding data: {e}")
            return serialized_data  # 返回原始字节
    
    def _recv_exact(self, sock: socket.socket, n: int) -> Optional[bytes]:
        """精确接收n字节数据"""
        data = b''
        while len(data) < n:
            try:
                chunk = sock.recv(n - len(data))
                if not chunk:
                    return None
                data += chunk
            except socket.timeout:
                continue
            except Exception:
                return None
        return data
    
    def _disconnect_client(self, client_socket: socket.socket, client_address: tuple):
        """断开客户端连接"""
        try:
            if client_socket in self.client_sockets:
                self.client_sockets.remove(client_socket)
            client_socket.close()
            print(f"🔌 Client {client_address} disconnected")
        except Exception as e:
            print(f"❌ Error disconnecting client: {e}")
    
    def send_to_client(self, client_socket: socket.socket, data: Any) -> bool:
        """向指定客户端发送数据"""
        return self._send_data(client_socket, data)
    
    def broadcast(self, data: Any) -> int:
        """向所有连接的客户端广播数据"""
        success_count = 0
        disconnected_clients = []
        
        for client_socket in self.client_sockets[:]:  # 创建副本以避免修改问题
            try:
                if self.send_to_client(client_socket, data):
                    success_count += 1
            except Exception:
                disconnected_clients.append(client_socket)
        
        # 清理断开连接的客户端
        for client_socket in disconnected_clients:
            if client_socket in self.client_sockets:
                self.client_sockets.remove(client_socket)
        
        return success_count
    
    def on_message(self, handler: Callable[[Any, socket.socket, tuple], None]):
        """注册消息处理器"""
        self.message_handlers.append(handler)
        return handler
    
    def stop(self):
        """停止服务器"""
        self.running = False
        for client_socket in self.client_sockets[:]:
            try:
                client_socket.close()
            except Exception:
                pass
        self.client_sockets.clear()
        
        if self.server_socket:
            try:
                self.server_socket.close()
            except Exception:
                pass
        
        print("🛑 Server stopped")


class SocketClient:
    """
    Socket客户端类，支持自动数据编码解码
    """
    
    def __init__(self, 
                 host: str = 'localhost', 
                 port: int = 12345,
                 serialization_method: SerializationMethod = SerializationMethod.PICKLE,
                 timeout: float = 10.0):
        """
        初始化客户端
        
        Args:
            host: 服务器主机地址
            port: 服务器端口号
            serialization_method: 序列化方法
            timeout: 连接超时时间
        """
        self.host = host
        self.port = port
        self.serialization_method = serialization_method
        self.timeout = timeout
        self.socket = None
        self.connected = False
        self.message_handlers = []
    
    def connect(self) -> bool:
        """连接到服务器"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(self.timeout)
            self.socket.connect((self.host, self.port))
            self.connected = True
            
            print(f"✅ Connected to server {self.host}:{self.port}")
            print(f"📦 Serialization method: {self.serialization_method.value}")
            
            # 启动接收消息的线程
            receive_thread = threading.Thread(target=self._receive_loop, daemon=True)
            receive_thread.start()
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to connect to server: {e}")
            self.connected = False
            return False
    
    def send(self, data: Any) -> bool:
        """发送数据到服务器"""
        if not self.connected or not self.socket:
            print("❌ Not connected to server")
            return False
        
        try:
            # 编码数据
            if self.serialization_method == SerializationMethod.PICKLE:
                serialized_data = pickle.dumps(data)
            elif self.serialization_method == SerializationMethod.JSON:
                if isinstance(data, (str, int, float, bool, list, dict, type(None))):
                    serialized_data = json.dumps(data).encode('utf-8')
                else:
                    serialized_data = pickle.dumps(data)
            elif self.serialization_method == SerializationMethod.STRING:
                if isinstance(data, str):
                    serialized_data = data.encode('utf-8')
                else:
                    serialized_data = str(data).encode('utf-8')
            else:
                serialized_data = pickle.dumps(data)
            
            # 发送数据长度前缀
            data_length = len(serialized_data)
            length_prefix = struct.pack('>I', data_length)
            self.socket.sendall(length_prefix + serialized_data)
            
            print(f"📤 Sent {data_length} bytes to server")
            return True
            
        except Exception as e:
            print(f"❌ Error sending data: {e}")
            self.connected = False
            return False
    
    def _receive_loop(self):
        """接收消息循环"""
        while self.connected:
            try:
                data = self._receive_data()
                if data is None:
                    break
                
                # 调用消息处理器
                for handler in self.message_handlers:
                    try:
                        handler(data)
                    except Exception as e:
                        print(f"❌ Error in message handler: {e}")
                        
            except socket.timeout:
                continue
            except Exception as e:
                if self.connected:
                    print(f"❌ Error receiving data: {e}")
                break
        
        self.connected = False
        print("🔌 Disconnected from server")
    
    def _receive_data(self) -> Any:
        """接收并解码数据"""
        if not self.connected or not self.socket:
            return None
        
        try:
            # 接收数据长度前缀
            length_data = self._recv_exact(4)
            if not length_data:
                return None
            
            data_length = struct.unpack('>I', length_data)[0]
            
            # 接收实际数据
            serialized_data = self._recv_exact(data_length)
            if not serialized_data:
                return None
            
            # 解码数据
            if self.serialization_method == SerializationMethod.PICKLE:
                return pickle.loads(serialized_data)
            elif self.serialization_method == SerializationMethod.JSON:
                return json.loads(serialized_data.decode('utf-8'))
            elif self.serialization_method == SerializationMethod.STRING:
                return serialized_data.decode('utf-8')
            else:
                return pickle.loads(serialized_data)
                
        except Exception as e:
            print(f"❌ Error receiving data: {e}")
            return None
    
    def _recv_exact(self, n: int) -> Optional[bytes]:
        """精确接收n字节数据"""
        data = b''
        while len(data) < n and self.connected:
            try:
                chunk = self.socket.recv(n - len(data))
                if not chunk:
                    return None
                data += chunk
            except socket.timeout:
                continue
            except Exception:
                return None
        return data
    
    def on_message(self, handler: Callable[[Any], None]):
        """注册消息处理器"""
        self.message_handlers.append(handler)
        return handler
    
    def disconnect(self):
        """断开连接"""
        self.connected = False
        if self.socket:
            try:
                self.socket.close()
            except Exception:
                pass
        print("🔌 Disconnected from server")