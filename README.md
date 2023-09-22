# AFDAS
大二实训人脸识别

## 技术选型
### 前端
- react——前端框架
- umi——对react的功能性封装
- ant design——组件库
- Webcam——摄像头调取
- Echarts/D3.js——可视化部分

### 后端
- 后端框架
	- flask框架
- 数据库
	-  flask_sqlalchemy 的 SQLAlchemy 接口框架
- 人脸识别
	- cv2：识别人脸处理


## 实现功能
- 登录界面
	- 员工/管理员登录
- 员工主界面
	- 人脸识别——打卡
	- 人脸更新——人脸录入
	- ........
- 管理员主界面
	- 打卡人员管理
		- 新增组
		- 删除组
		- 新增新员工
		- 删除旧员工——逻辑删除
	- 打卡数据可视化
	- 提醒警告？——邮箱/短信

## 日程规划

### Day1：6.25

#### 人脸录入、识别功能实现——ok
Haar特征分类器是一种基于Haar小波变换的特征提取方法。它可以用于图像处理中的对象检测和识别。Haar分类器通过将图像中的不同区域与预定义的Haar特征进行比较，来检测物体的存在。这些Haar特征通常是由边缘、线和矩形组成的，可以描述图像中不同区域的亮度和颜色特征。在训练过程中，分类器会学习哪些Haar特征对于识别特定对象是最有用的。Haar特征分类器被广泛应用于人脸检测、行人检测等领域。

[参考网站](https://github.com/Chando0185/face_recognition_project/tree/main)

```python
import os
import cv2
import pickle
import numpy as np


def add_faces(name,photos):
    facedetect=cv2.CascadeClassifier('./static/model/haarcascade_frontalface_default.xml')
    faces_data = []
    i = 0
    for i in range(len(photos)):
        # cv2读取图像
        frame = photos[i]
        # 转换为灰度图像
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 人脸检测
        faces = facedetect.detectMultiScale(gray, 1.3, 5)
        # 人脸框图位置
        for (x, y, w, h) in faces:
            crop_img = frame[y:y+h, x:x+w, :]
            resized_img = cv2.resize(crop_img, (50, 50))
            if len(faces_data) <=20:
                faces_data.append(resized_img)
        if len(faces_data) == 20:
            break

    # 转化为numpy数组特征数据
    faces_data = np.asarray(faces_data)
    faces_data = faces_data.reshape(len(faces_data), -1)
    # 保存特征名字
    # 如果文件夹不存在，则创建文件夹
    if not os.path.exists('./static/data/'):
        os.makedirs('./static/data/')

    if 'names.pkl' not in os.listdir('./static/data/'):
            names = [name] * len(faces_data)
            with open('./static/data/names.pkl', 'wb') as f:
                pickle.dump(names, f)
    else:
        with open('./static/data/names.pkl', 'rb') as f:
            names = pickle.load(f)
        names = names + [name] * len(faces_data)
        with open('./static/data/names.pkl', 'wb') as f:
            pickle.dump(names, f)

    # 保存特征数据
    if 'faces_data.pkl' not in os.listdir('./static/data/'):
        with open('./static/data/faces_data.pkl', 'wb') as f:
            pickle.dump(faces_data, f)
    else:
        with open('./static/data/faces_data.pkl', 'rb') as f:
            faces = pickle.load(f)
        faces = np.append(faces, faces_data, axis=0)
        with open('./static/data/faces_data.pkl', 'wb') as f:
            pickle.dump(faces, f)
    return True


def detect_faces(photos):
    from sklearn.neighbors import KNeighborsClassifier
    import cv2
    import pickle
    import numpy as np
    import os
    import csv
    import time
    from datetime import datetime
    attendance=[]
    facedetect = cv2.CascadeClassifier('./static/model/haarcascade_frontalface_default.xml')
    # 加载数据
    with open('./static/data/names.pkl', 'rb') as w:
        LABELS = pickle.load(w)
    with open('./static/data/faces_data.pkl', 'rb') as f:
        FACES = pickle.load(f)

    print('Shape of Faces matrix --> ', FACES.shape)

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(FACES, LABELS)
    for photo in photos:
        frame =photo
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facedetect.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            crop_img = frame[y:y+h, x:x+w, :]
            resized_img = cv2.resize(crop_img, (50, 50)).flatten().reshape(1, -1)
            output = knn.predict(resized_img)
            attendance.append(str(output[0]))
    return attendance

# 摄像头保存照片20张到./static/data下面
def get_faces(num):
    # 指定保存图像的目录
    save_dir = './static/pictures'
    # 创建保存图像的目录（如果不存在）
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # 使用默认摄像头
    cap = cv2.VideoCapture(0)
    # 获取并保存20张图像
    for i in range(num):
        # 从摄像头读取图像
        ret, frame = cap.read()
        # 生成图像文件名
        filename = f'image_{i+1}.jpg'
        # 保存图像
        cv2.imwrite(os.path.join(save_dir, filename), frame)
        # 显示保存的文件路径
        print(f'Saved: {os.path.join(save_dir, filename)}')
    # 释放摄像头资源
    print("获取图片完成")
    cap.release()

def test(num, name):
    print("test")
    # 选择拍摄多少张图片
    get_faces(num)
    # 读取图片
    photos = []
    for i in range(num):
        photos.append(f'./static/pictures/image_{i+1}.jpg')
    add_faces(name,photos)
    # 识别人脸
    result = detect_faces(photos)
    # 找到result里面出现次数最多的
    result = max(result, key=result.count)
    if result == name:
        print("识别成功")
    else:
        print("识别失败")
```

#### 项目初始化——ok
##### umi框架初始化
[详情](https://ant.design/docs/react/use-with-umi-cn)
```js
├── config/
    ├── config.js                  // umi 配置，同 .umirc.js，二选一
├── dist/                          // 默认的 build 输出目录
├── mock/                          // mock 文件所在目录，基于 express
├── public/                        // 全局相对路径文件
└── src/                           // 源码目录，可选
    ├── assets/                    // 静态文件
    ├── components/                // 全局共用组件
    ├── layouts/index.js           // 全局入口文件
    ├── models/                    // 全局models文件，存放全局共用数据store
    ├── pages/                     // 页面目录，业务组件
        ├── .umi/                  // dev 临时目录，需添加到 .gitignore
        ├── .umi-production/       // build 临时目录，会自动删除
        ├── index/                 // 首页模块
        ├── manager/               // 管理端模块
            ├── components/        // 管理端-局部公共组件
            ├── models/            // 管理端-局部models，存放manager的store
            ├── services/          // 管理端-局部services，存放manager的接口
            ├── index.js           // 业务组件index
            ├── page.js            // 业务组件page
            ├── _layout.js         // 局部入口文件
        ├── 404.js                 // 404 页面
    ├── services/                  // 全局services文件，存放全局公共接口
    ├── utils/                     // 全局工具类
    ├── global.css                 // 约定的全局样式文件，自动引入，也可以用 global.less
    ├── global.js                  // 约定的全局Js文件，自动引入，可以在这里加入 polyfill
    ├── app.js                     // 运行时配置文件
├── .umirc.js                      // umi 配置，同 config/config.js，二选一
├── .env                           // 环境变量
└── package.json
```

##### flask框架demo初始化+数据库初始化
[基本框架demo](https://github.com/Mercurius14/Service_Outsourcing-/blob/master/register.py)
**Flask初始化**
```python
app = Flask(__name__)

app.config['JSON_AS_ASCII'] = False # jsonify转变格式的时候不会转变为unicode编码格式，unicode编码格式无法直接看到汉字

app.config['JSONIFY_MIMETYPE'] = "application/json;charset=utf-8" # 指定浏览器渲染的文件类型，和解码格式

# 利用flask_cors设置跨域
CORS(app, supports_credentials=True)

# 配置表单密令
app.secret_key = '\xfe{\xa9\n\x1b0\x16\xcfF\xb103\x9d)\xdf\xfd\xab\xd8\x9b\xbf\xf2\xf5\xb0\x86'

app.config.from_object(__name__) # 应用配置
```

**sqlite数据库初始化**
1. 基础信息配置
```python
#数据库配置
DIALCT = "mysql"
DRIVER = "pymysql"
USERNAME = "root"
PASSWORD = "*8888"
HOST = "127.0.0.1"
PORT = "3306"
DATABASE = "facedetect"
DB_URI = "{}+{}://{}:{}@{}:{}/{}?charset=utf8".format(DIALCT,DRIVER,USERNAME,PASSWORD,HOST,PORT,DATABASE)
app.config["SQLALCHEMY_DATABASE_URI"] = DB_URI

# 实例化所需模块
db = SQLAlchemy(app)
# 用于管理数据库迁移
# manager = Manager(app)
# migrate = Migrate(app, db)
# manager.add_command('db', MigrateCommand)
```
2. 数据库——还需修改
- 员工信息表——员工
- 人脸记录表
- 签到考勤表
```python
class Group(db.Model):
    __tablename__ ="group"
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    groupName = db.Column(db.String(255),nullable=False)


# 学生信息表
class Student(db.Model):
    __tablename__ = "student"
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    name = db.Column(db.String(255))
    # 学号，不允许相同
    stuId = db.Column(db.String(255),nullable=False,unique=True)
    # stuAccount = db.Column(db.String(255),nullable=False)
    stuPassword = db.Column(db.String(255),nullable=False)
    stuGroup = db.Column(db.String(255), db.ForeignKey('group.groupName'))
    # phone = db.Column(db.String(255),default='')
    # email = db.Column(db.String(255),default='')
    StuStatus = db.Column(db.Integer,default=0) # 0是正常，1是异常
    createTime = db.Column(db.DateTime, default=datetime.now())
    updateTime = db.Column(db.DateTime, default=datetime.now())
    isRole = db.Column(db.Integer, default=0) # 0是普通学生，1是管理员
    isDelete = db.Column(db.Integer, default=0) # 0是未删除，1是已删除
    # 一对多关系
    # 一个学生可以有多个考勤记录
    attendances = db.relationship('Attendance', backref='student', lazy='dynamic')
    # 一个学生可以有多个人脸记录
    faceRecords = db.relationship('FaceRecord', backref='student', lazy='dynamic')


# 人脸记录表
class FaceRecord(db.Model):
    __tablename__ = "faceRecord"
    id = db.Column(db.Integer, primary_key=True)
    stuId = db.Column(db.String(255), db.ForeignKey('student.stuId'))
    stuName = db.Column(db.String(255),nullable=False)
    faceEncoding = db.Column(db.Text,nullable=False)
    createTime = db.Column(db.DateTime, default=datetime.now())
    updateTime = db.Column(db.DateTime, default=datetime.now())
    isDelete = db.Column(db.Integer, default=0) # 0是未删除，1是已删除


# 考勤表——记录学生的考勤信息
class Attendance(db.Model):
    __tablename__ = "attendance"
    id = db.Column(db.Integer, primary_key=True)
    stuId = db.Column(db.String(255), db.ForeignKey('student.stuId'))
    stuName = db.Column(db.String(255),nullable=False)
    # 0是上班打卡，1是下班打卡
    attendanceType = db.Column(db.Integer)
    # 0是正常，1是迟到，2是早退
    attendanceStatus = db.Column(db.Integer)

    attendanceTime = db.Column(db.DateTime,default=datetime.now()) 
```
 
```sql
CREATE TABLE `group` (
  `id` INT PRIMARY KEY AUTO_INCREMENT,
  `groupName` VARCHAR(255) NOT NULL
);

CREATE TABLE `student` (
  `id` INT PRIMARY KEY AUTO_INCREMENT,
  `name` VARCHAR(255),
  `stuId` VARCHAR(255) NOT NULL UNIQUE,
  `stuPassword` VARCHAR(255) NOT NULL,
  `stuGroup` VARCHAR(255),
  `stuStatus` INT DEFAULT 0,
  `createTime` DATETIME DEFAULT CURRENT_TIMESTAMP,
  `updateTime` DATETIME DEFAULT CURRENT_TIMESTAMP,
  `isRole` INT DEFAULT 0,
  `isDelete` INT DEFAULT 0,
  FOREIGN KEY (`stuGroup`) REFERENCES `group`(`groupName`)
);

CREATE TABLE `faceRecord` (
  `id` INT PRIMARY KEY,
  `stuId` VARCHAR(255),
  `stuName` VARCHAR(255) NOT NULL,
  `faceEncoding` TEXT NOT NULL,
  `createTime` DATETIME DEFAULT CURRENT_TIMESTAMP,
  `updateTime` DATETIME DEFAULT CURRENT_TIMESTAMP,
  `isDelete` INT DEFAULT 0,
  FOREIGN KEY (`stuId`) REFERENCES `student`(`stuId`)
);

CREATE TABLE `attendance` (
  `id` INT PRIMARY KEY,
  `stuId` VARCHAR(255),
  `stuName` VARCHAR(255) NOT NULL,
  `attendanceType` INT,
  `attendanceStatus` INT,
  `attendanceTime` DATETIME DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (`stuId`) REFERENCES `student`(`stuId`)
);

```


### Day2：6.26

#### 前端人脸识别调取、录入照片传输方法查找
**利用Webcam组件**
```ts
const webcamRef = useRef<Webcam>(null);
//.....
const imageSrc = (webcamRef.current as Webcam)?.getScreenshot();

//...
<Webcam  
	ref={webcamRef}  
	mirrored={true}  
	style={{ width: '100%', height: '100%' }}  
/>
```
**前端**
Data URI转换为Blob对象
```ts
const dataURItoBlob = (dataURI: string) => {  
	const byteString = atob(dataURI.split(',')[1]);  
	const mimeString = dataURI.split(',')[0].split(':')[1].split(';')[0];  
	const ab = new ArrayBuffer(byteString.length);  
	const ia = new Uint8Array(ab);  
	for (let i = 0; i < byteString.length; i++) {  
		ia[i] = byteString.charCodeAt(i);  
	}  
	return new Blob([ab], { type: mimeString });  
};
```
**后端**
需要进行解码
```python
file = request.files[image_key]# 进行解码
image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
```
#### 员工识别界面设计——需要更改
```ts
import React, { useState, useRef } from 'react';
import Webcam from 'react-webcam';
import { Button, Modal } from 'antd';
import axios from "axios";

const DetectCom = () => {
    // 设置摄像头
    const webcamRef = useRef<Webcam>(null);
    // 是否显示摄像头
    const [showWebcam, setShowWebcam] = useState(false);
    // 拍摄的照片
    const [capturedImages, setCapturedImages] = useState<string[]>([]);
    // 是否显示成功的模态框
    const [showSuccessModal, setShowSuccessModal] = useState(false);
    // 是否显示失败的模态框
    const [showErrorModal, setShowErrorModal] = useState(false);

    const handleStartRecognition = () => {
        setShowWebcam(true);
        captureImages();
    };

    const captureImages = () => {
        const images: string[] = [];

        const captureImage = () => {
            if (images.length < 10) {
                const imageSrc = (webcamRef.current as Webcam)?.getScreenshot();
                //console.log(imageSrc)
                if (imageSrc) {
                    //console.log("拍摄一张照片")
                    images.push(imageSrc);
                    setCapturedImages(images);
                    if (images.length === 50) {
                        //console.log("十张照片够了")
                        // 测试
                        // setShowSuccessModal(true);
                        // setShowWebcam(false);
                        processImages(images);

                    } else {
                        setTimeout(captureImage, 20); // 每秒拍摄一张照片
                    }
                }
                else{
                    setTimeout(captureImage, 100); // 每秒拍摄一张照片
                }
            }
        };
        captureImage();
    };

    const processImages = (images: string[]) => {
        const formData = new FormData();
        images.forEach((image, index) => {
            const blob = dataURItoBlob(image);
            formData.append('image' + index, blob, 'image' + index + '.jpg');
        });

        axios
            .post('http://localhost:8098/detect', formData)
            .then((response) => {
                const msg = response.data.msg;
                // console.log(msg);
                if (msg === "1") {
                    setShowSuccessModal(true);
                    setShowWebcam(false);
                } else {
                    setShowErrorModal(true);
                    setShowWebcam(false);
                }
            })
            .catch((error) => {
                console.error('Error:', error);
                setShowErrorModal(true);
                setShowWebcam(false);
            });
    };

    // 辅助函数：将Data URI转换为Blob对象
    const dataURItoBlob = (dataURI: string) => {
        const byteString = atob(dataURI.split(',')[1]);
        const mimeString = dataURI.split(',')[0].split(':')[1].split(';')[0];
        const ab = new ArrayBuffer(byteString.length);
        const ia = new Uint8Array(ab);
        for (let i = 0; i < byteString.length; i++) {
            ia[i] = byteString.charCodeAt(i);
        }
        return new Blob([ab], { type: mimeString });
    };

    const handleSuccessModalOk = () => {
        setShowSuccessModal(false);
    };

    const handleErrorModalOk = () => {
        setShowErrorModal(false);
    };

    return (
        <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100vh' }}>
            <div style={{ position: 'relative', width: '300px', height: '225px' }}>
                {showWebcam && (
                    <div
                        style={{
                            position: 'absolute',
                            border: '2px solid #ccc',
                            top: '50%',
                            left: '50%',
                            transform: 'translate(-50%, -50%)',
                            width: '100%',
                            height: '100%',
                        }}
                    >
                        {/* 使用Webcam组件显示摄像头 */}
                        <Webcam
                            ref={webcamRef}
                            mirrored={true}
                            style={{ width: '100%', height: '100%' }}
                        />
                    </div>
                )}
                {!showWebcam && (
                    <div
                        style={{
                            position: 'absolute',
                            border: '2px solid #ccc',
                            top: '50%',
                            left: '50%',
                            transform: 'translate(-50%, -50%)',
                            width: '100%',
                            height: '100%',
                            backgroundColor: '#f5f5f5',
                        }}
                    />
                )}
                <div
                    style={{
                        position: 'absolute',
                        top: '50%',
                        left: '50%',
                        transform: 'translate(-50%, -50%)',
                        display: 'flex',
                        flexDirection: 'column',
                        alignItems: 'center',
                    }}
                >
                    {!showWebcam && <Button onClick={handleStartRecognition}>开始识别</Button>}
                </div>
            </div>

            <Modal
                title="打卡成功"
                visible={showSuccessModal}
                onOk={handleSuccessModalOk}
            >
                <p>打卡成功</p>
            </Modal>

            <Modal
                title="打卡识别失败"
                visible={showErrorModal}
                onOk={handleErrorModalOk}
            >
                <p>打卡失败</p>
            </Modal>
        </div>
    );
};
export default DetectCom;

```

#### 员工录入界面设计——需要更改
```ts
import React, { useState, useRef } from 'react';
import Webcam from 'react-webcam';
import { Button, Modal } from 'antd';
import axios from "axios";

const UploadFace = () => {
    // 设置摄像头
    const webcamRef = useRef<Webcam>(null);
    // 是否显示摄像头
    const [showWebcam, setShowWebcam] = useState(false);
    // 拍摄的照片
    const [capturedImages, setCapturedImages] = useState<string[]>([]);
    // 是否显示成功的模态框
    const [showSuccessModal, setShowSuccessModal] = useState(false);
    // 是否显示失败的模态框
    const [showErrorModal, setShowErrorModal] = useState(false);

    const handleStartRecognition = () => {
        setShowWebcam(true);
        captureImages();
    };

    const captureImages = () => {
        const images: string[] = [];

        const captureImage = () => {
            if (images.length < 10) {
                const imageSrc = (webcamRef.current as Webcam)?.getScreenshot();
                // console.log(imageSrc)
                if (imageSrc) {
                    // console.log("拍摄一张照片")
                    images.push(imageSrc);
                    setCapturedImages(images);
                    if (images.length === 50) {
                        //console.log("十张照片够了")
                        // 测试
                        // setShowSuccessModal(true);
                        // setShowWebcam(false);

                        processImages(images,"test");

                    } else {
                        setTimeout(captureImage, 20); // 每0.02秒拍摄一张照片
                    }
                }
                else{
                    setTimeout(captureImage, 100); // 每秒拍摄一张照片
                }
            }
        };
        captureImage();
    };

    const processImages = (images: string[], name: string) => {
        // console.log("开始传输照片拉", images);

        const formData = new FormData();
        images.forEach((image, index) => {
            const blob = dataURItoBlob(image);
            formData.append('image' + index, blob, 'image' + index + '.jpg');
        });
        formData.append('name', name);  // 添加name参数

        axios
            .post('http://localhost:8098/upload', formData)
            .then((response) => {
                const msg = response.data.msg;
                // console.log(msg);
                if (msg === "1") {
                    setShowSuccessModal(true);
                    setShowWebcam(false);
                } else {
                    setShowErrorModal(true);
                    setShowWebcam(false);
                }
            })
            .catch((error) => {
                console.error('Error:', error);
                setShowErrorModal(true);
                setShowWebcam(false);
            });
    };

    // 辅助函数：将Data URI转换为Blob对象
    const dataURItoBlob = (dataURI: string) => {
        const byteString = atob(dataURI.split(',')[1]);
        const mimeString = dataURI.split(',')[0].split(':')[1].split(';')[0];
        const ab = new ArrayBuffer(byteString.length);
        const ia = new Uint8Array(ab);
        for (let i = 0; i < byteString.length; i++) {
            ia[i] = byteString.charCodeAt(i);
        }
        return new Blob([ab], { type: mimeString });
    };

    const handleSuccessModalOk = () => {
        setShowSuccessModal(false);
    };

    const handleErrorModalOk = () => {
        setShowErrorModal(false);
    };

    return (
        <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100vh' }}>
            <div style={{ position: 'relative', width: '300px', height: '225px' }}>
                {showWebcam && (
                    <div
                        style={{
                            position: 'absolute',
                            border: '2px solid #ccc',
                            top: '50%',
                            left: '50%',
                            transform: 'translate(-50%, -50%)',
                            width: '100%',
                            height: '100%',
                        }}
                    >
                        {/* 使用Webcam组件显示摄像头 */}
                        <Webcam
                            ref={webcamRef}
                            mirrored={true}
                            style={{ width: '100%', height: '100%' }}
                        />
                    </div>
                )}
                {!showWebcam && (
                    <div
                        style={{
                            position: 'absolute',
                            border: '2px solid #ccc',
                            top: '50%',
                            left: '50%',
                            transform: 'translate(-50%, -50%)',
                            width: '100%',
                            height: '100%',
                            backgroundColor: '#f5f5f5',
                        }}
                    />
                )}
                <div
                    style={{
                        position: 'absolute',
                        top: '50%',
                        left: '50%',
                        transform: 'translate(-50%, -50%)',
                        display: 'flex',
                        flexDirection: 'column',
                        alignItems: 'center',
                    }}
                >
                    {!showWebcam && <Button onClick={handleStartRecognition}>开始录入</Button>}
                </div>
            </div>

            <Modal
                title="录入成功"
                visible={showSuccessModal}
                onOk={handleSuccessModalOk}
            >
                <p>录入成功</p>
            </Modal>

            <Modal
                title="录入识别失败"
                visible={showErrorModal}
                onOk={handleErrorModalOk}
            >
                <p>录入失败</p>
            </Modal>
        </div>
    );
};

export default UploadFace;

```

### Day3：6.27

#### 后端登录设计
##### 密码加密——需要添加

##### 用户脱敏
避免用户密码等信息泄露
```python
def infoDesensitization(student):
    # 返回给前端的学生信息
    return_student = {}
    return_student['stuId'] = student['stuId']
    return_student['stuName'] = student['stuName']
    return_student['stuGroup'] = student['stuGroup']
    return_student['isRole'] = student['isRole']
    return return_student
```
##### 枚举类状态码返回
```python
from enum import Enum

# 枚举类定义返回状态码
class StatusCode(Enum):
    # 登录失败
    LOGIN_FAIL = 1101
    # 登录成功
    LOGIN_SUCCESS = 1001
    # 注册失败
    REGISTER_FAIL = 1102
    # 注册成功
    REGISTER_SUCCESS = 1002
    # 未登录
    NOT_LOGIN = 1103
    # 未注册
    NOT_REGISTER = 1104
    # 已登录
    ALREADY_LOGIN = 1003
	# 识别成功
    DETECT_SUCCESS = 1004
    # 识别失败
    DETECT_FAIL = 1105
    # 识别错误
    DETECT_ERROR = 1106
    # 录入成功
    UPLOAD_SUCCESS = 1005
    # 录入失败
    UPLOAD_FAIL = 1107
    # 未录入
    NOT_ADD = 1108
    # 未知错误
    UNKNOWN_ERROR = 1109
```
##### session保存用户信息——没有实现有BUG
```python
		# ......
		student = Student.query.filter_by(stuId=stuId).first()
        if student:
            if student.stuPassword == stuPassword:
                # 将用户信息存入session
                session['student'] = student.to_dict()
                data = session.get('student')
                return_data = infoDesensitization(data)
				# ......


@app.route('/api/checkSession')
def check_session():
    # 在这里编写检查会话状态的逻辑
    if session.get('student'):
        stu_data = session.get('student')
        stu_data = infoDesensitization(stu_data)
        return jsonify({'msg': '已经登录', 'status': StatusCode.ALREADY_LOGIN, "data": stu_data})
    else:
        return jsonify({'msg': '未登录', 'status': StatusCode.NOT_LOGIN, "data": {}})
```

#### 前端设计
##### 上下文设计——全局变量用户信息

```ts
import React, { createContext, useState } from 'react';

// 学生sesstion信息存储
type StudentContextType = {
    studentData: any;
    setStudentData: (data: any) => void;
};

// 学生信息上下文提供者的属性类型

interface StudentContextProviderProps {
    children: React.ReactNode;
}
// 创建全局上下文，并提供初始值

export const StudentContext = createContext<StudentContextType>({
    studentData: null,
    setStudentData: () => {},
});

// 全局上下文提供者组件
export const StudentContextProvider: React.FC<StudentContextProviderProps> = ({ children }) => {
    const [studentData, setStudentData] = useState(null);

    return (
        <StudentContext.Provider value={{ studentData, setStudentData }}>
            {children}
        </StudentContext.Provider>
    );
};

```

##### APi调用集中管理
将所有的前后端端口调用函数集中到service/api.ts

#### 登录界面设计
利用ProCompnents的登录界面
[官网](https://procomponents.ant.design/components/login-form)

##### 图片添加
```ts
// 导入图片  
import backgroundImage from '@/assets/background.jpg';  
const logoImage = require('@/assets/squai.png');
// 背景
<div style={{  
	backgroundImage: `url(${backgroundImage})`,  
	backgroundSize: 'cover',  
	backgroundPosition: 'center',  
	height: '100vh', // 设置元素高度为100vh，即视口的高度  
	backgroundColor: 'white' }}>
// logo
<LoginForm  
	logo={logoImage}  
	title="AI人脸识别打卡系统"  
	subTitle="大二暑期实训"  
	onFinish={handleLogin}  
	// 设置背景  
	// style={{  
	// backgroundImage: 'url(https://th.bing.com/th/id/OIP.a9w8HIGtum3eR9zfKHRKgAHaEK?w=203&h=114&c=7&r=0&o=5&dpr=1.5&pid=1.7)',  
// }}  
>

```

### Day4、5：6.28、29

#### 前端返回值结构化

```python
class ReturnData:
    def __init__(self, msg, status, data):
        self.msg = msg
        self.status = status
        self.data = data

    def to_dict(self):
        return {
            'msg': self.msg,
            'status': self.status,
            'data': self.data
        }
```

#### 学生界面完善
##### 路由传递参数

利用url的参数进行传递
>路由转换
```ts
history.push(`/employee?name=${response.data?.data.name}`);
```
>获取路由中name参数
```ts
// 获取url中的参数  
const location = useLocation();  
const params = new URLSearchParams(location.search);  
const name = params.get('name');  
console.log("姓名",name)
```


##### 打卡时间判断
```text
# 打卡时间判断
# input: None
# output: attendanceType: int ,    0是上班打卡，1是下班打卡
#         attendanceStatus: int    0是正常，1是迟到，2是早退
```

#### 管理员页面加载阶段获取学生信息

##### 界面加载界面获取信息
```ts
useEffect(() => {
        // 在组件渲染之前发送GET请求
        const fetchData = async () => {
            try {
                const response = await getStudents();
                console.log("response", response);
                setStudentData(response?.data);
            } catch (error) {
                console.error(error);
            }
        };

        fetchData();

        return () => {
            // 在组件卸载时执行一些清理操作
        };
    }, []);


// 获取学生信息

export async function getStudents() {
    try {
        // 发起网络请求获取学生数据
        const response = await axios.get('http://localhost:8098/students');
        return response.data; // 返回后端返回的数据
    } catch (error) {
        // 处理错误情况
        console.error(error);
        // 显示错误信息
        // 示例：message.error('获取学生数据失败，请重试！');
    }
```

##### 数据上下文分享
详情看[[#上下文设计——全局变量用户信息]]
```ts
    const { studentData, setStudentData } = useContext(StudentContext);
```
### Day6：6.30
#### 后端设计

##### 需要返回的数据信息——需要修改
###### 总体信息
学生信息
- stuId
- stuPassword
- stuGroup
- StuStatus
- createTime
- updateTime


#### 管理员界面设计
##### 前端
- 展示——利用可编辑表格组件
- 组别添加
- 学生添加
- 学生删除
- 学生修改



### Day7：7.1
#### 组别和学生逻辑完善
##### 新增、修改、删除学生——组别变化
修改对应的handle函数
#### 后端数据库对应增删改查处理


### Day8：7.2

#### 考勤信息展示
##### 学生端
时间轴组件
##### 管理员端
表格组件

### Day9： 7.3
#### 学生信息修改界面
利用Description组件和气泡框组件

#### 人脸录入和识别逻辑完善
识别之前需要先录入

### Day10,11：7.4,5
#### 增删改查UI美化
添加了搜索框可以按学号、进行搜索
#### 添加管理员可视化
利用echarts组件
- 柱状图——最近一周的柱状图
	- 正常
	- 早退
	- 迟到
	- 未考勤
- 扇形图——今天的考勤情况——未考勤、正常、迟到和早退

### Day12~：后续完善工作
#### 七天图表图

#### session保存
```ts
import axios from 'axios';

axios.get('/api/get_data', { withCredentials: true })
  .then(response => {
    console.log(response.data);
  })
  .catch(error => {
    console.error(error);
  });

```
当您在前端使用 `withCredentials: true` 选项时，浏览器会自动将会话凭据（如 cookie）添加到请求的头部中，然后将请求发送到后端。在后端接收到请求后，如果已正确设置会话管理器（如使用了 `express-session` 中间件），会话数据将被存储在服务器端的会话存储中，并与该会话相关联。
### 答辩
#### 模型讲解

#### 特征散点图区分



```
通过参与课程设计，我对AI人脸识别打卡系统有了更深入的了解，并获得了以下感悟和学习：

1.  综合技术应用：在这个课程设计中，我学会了如何将多种技术和框架结合使用，如React、umi框架、ant design、echarts和flask等。通过整合这些技术，我能够构建一个完整的AI人脸识别打卡系统，并了解了它们在实际项目中的应用。
    
2.  人脸识别技术：课程设计的重点是人脸识别功能的实现。通过学习和实践，我了解了人脸识别算法的原理和常见的实现方法，如使用深度学习模型进行人脸特征提取和匹配。这让我对人脸识别技术有了更深入的了解，并能够应用到其他相关领域中。
    
3.  用户界面设计：在课程设计中，我学到了如何使用React和ant design等工具来设计用户友好的界面。这包括布局设计、交互设计和样式美化等方面。通过实践，我学会了如何考虑用户体验和界面美观性，使系统更易用和吸引人。
    
4.  安全性和隐私保护：课程设计中的人脸识别系统涉及到用户的个人隐私和敏感信息，因此我对系统的安全性和隐私保护有了更深刻的认识。我了解了数据加密、访问控制和数据泄露预防等安全措施的重要性，并学会了如何在设计中考虑这些因素。
    

通过这个课程设计，我不仅掌握了技术和工具的使用，还加深了对人脸识别技术和系统设计的理解。我学会了将不同的技术和概念整合到一个实际项目中，并意识到了系统开发中需要考虑的安全性和用户体验等方面。这些知识和经验将对我今后在人工智能和系统开发领域的学习和实践中有很大帮助。
```
