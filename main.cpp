#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <random> // 用于SSAO随机性
#include <string>
#include <vector>
#include <filesystem>
#include <numbers>

#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

// // 定义PI，确保其可用性
// #ifndef M_PI
// #define M_PI 3.14159265358979323846f
// #endif

// ========================
// 基础数学结构: Vec3, Vec2, Vec4, Mat4
// ========================

// 三维向量结构
struct Vec3 {
  float x, y, z;

  // 默认构造函数，初始化为零
  Vec3() : x(0.0f), y(0.0f), z(0.0f) {}
  // 带参数构造函数
  Vec3(float _x, float _y, float _z) : x(_x), y(_y), z(_z) {}

  // 向量加法
  Vec3 operator+(const Vec3 &o) const { return {x + o.x, y + o.y, z + o.z}; }
  // 向量减法
  Vec3 operator-(const Vec3 &o) const { return {x - o.x, y - o.y, z - o.z}; }
  // 向量与标量乘法
  Vec3 operator*(float s) const { return {x * s, y * s, z * s}; }
  // 向量与标量除法
  Vec3 operator/(float s) const {
    if (s == 0)
      return {0, 0, 0}; // 避免除以零
    return {x / s, y / s, z / s};
  }
  // 向量取反
  Vec3 operator-() const { return {-x, -y, -z}; }

  // 分量乘法 (Hadamard product)
  Vec3 operator*(const Vec3 &o) const { return {x * o.x, y * o.y, z * o.z}; }

  // 输出流重载，用于打印Vec3
  friend std::ostream &operator<<(std::ostream &os, const Vec3 &v) {
    os << "Vec3(" << v.x << ", " << v.y << ", " << v.z << ")";
    return os;
  }

  friend std::istream &operator>>(std::istream &is, Vec3 &v) {
    is >> v.x >> v.y >> v.z;
    return is;
  }

  static Vec3 lerp(const Vec3 &a, const Vec3 &b, float t) {
    return {a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t,
            a.z + (b.z - a.z) * t};
  }

  // 点积
  float dot(const Vec3 &o) const { return x * o.x + y * o.y + z * o.z; }
  // 叉积
  Vec3 cross(const Vec3 &o) const {
    return {y * o.z - z * o.y, z * o.x - x * o.z, x * o.y - y * o.x};
  }
  // 向量长度
  float length() const { return std::sqrt(x * x + y * y + z * z); }
  // 向量归一化
  Vec3 normalize() const {
    float l = length();
    if (l == 0)
      return {0, 0, 0}; // 避免除以零
    return {x / l, y / l, z / l};
  }
};
// 二维向量结构
struct Vec2 {
  float u, v;

  // 默认构造函数
  Vec2() : u(0.0f), v(0.0f) {}
  // 带参数构造函数
  Vec2(float _u, float _v) : u(_u), v(_v) {}

  Vec2 operator=(const Vec2 &o) {
    u = o.u;
    v = o.v;
    return *this;
  }

  // 向量与标量乘法
  Vec2 operator*(float s) const { return {u * s, v * s}; }
  // 向量加法
  Vec2 operator+(const Vec2 &o) const { return {u + o.u, v + o.v}; }

  // 向量减法
  Vec2 operator-(const Vec2 &o) const { return {u - o.u, v - o.v}; }

  Vec2 operator-=(const Vec2 &o) {
    u -= o.u;
    v -= o.v;
    return *this;
  }

  // 除法
  Vec2 operator/(float s) const {
    if (s == 0)
      return {0, 0}; // 避免除以零
    return {u / s, v / s};
  }

  friend std::ostream &operator<<(std::ostream &os, const Vec2 &v) {
    os << "Vec2(" << v.u << ", " << v.v << ")";
    return os;
  }
};

// 四维向量结构 (用于齐次坐标)
struct Vec4 {
  float x, y, z, w;

  // 默认构造函数
  Vec4() : x(0.0f), y(0.0f), z(0.0f), w(0.0f) {}
  // 带参数构造函数
  Vec4(float _x, float _y, float _z, float _w) : x(_x), y(_y), z(_z), w(_w) {}

  // 向量与标量除法
  Vec4 operator/(float s) const {
    if (s == 0)
      return {0, 0, 0, 0}; // 避免除以零
    return {x / s, y / s, z / s, w / s};
  }

  // 向量归一化
  Vec4 normalize() const {
    float l = std::sqrt(x * x + y * y + z * z + w * w);
    if (l == 0)
      return {0, 0, 0, 0};
    return {x / l, y / l, z / l, w / l};
  }
  // 转换为Vec3 (通常用于齐次坐标转换后丢弃w)
  operator Vec3() const { return Vec3(x, y, z); }
};

struct Mat4 {
  float m[4][4];

  Mat4() {
    for (int i = 0; i < 4; ++i) {
      for (int j = 0; j < 4; ++j) {
        m[i][j] = 0.0f;
      }
    }
  }

  // 静态方法，创建单位矩阵
  static Mat4 identity() {
    Mat4 res;
    res.m[0][0] = 1.0f;
    res.m[1][1] = 1.0f;
    res.m[2][2] = 1.0f;
    res.m[3][3] = 1.0f;
    return res;
  }

  // 矩阵乘法
  Mat4 multiply(const Mat4 &o) const {
    Mat4 res;
    for (int i = 0; i < 4; ++i) {
      for (int j = 0; j < 4; ++j) {
        for (int k = 0; k < 4; ++k) {
          res.m[i][j] += m[i][k] * o.m[k][j];
        }
      }
    }
    return res;
  }

  // 矩阵与Vec4相乘
  Vec4 multiply(const Vec4 &v) const {
    return {m[0][0] * v.x + m[0][1] * v.y + m[0][2] * v.z + m[0][3] * v.w,
            m[1][0] * v.x + m[1][1] * v.y + m[1][2] * v.z + m[1][3] * v.w,
            m[2][0] * v.x + m[2][1] * v.y + m[2][2] * v.z + m[2][3] * v.w,
            m[3][0] * v.x + m[3][1] * v.y + m[3][2] * v.z + m[3][3] * v.w};
  }

  // 矩阵与矩阵乘法 (R = A * B)
  Mat4 operator*(const Mat4 &o) const {
    Mat4 r;
    for (int i = 0; i < 4; ++i) {   // 结果矩阵的行
      for (int j = 0; j < 4; ++j) { // 结果矩阵的列
        r.m[i][j] = 0;
        for (int k = 0; k < 4; ++k) {
          r.m[i][j] += m[i][k] * o.m[k][j];
        }
      }
    }
    return r;
  }

  // 获取转置矩阵
  Mat4 transpose() const {
    Mat4 res;
    for (int i = 0; i < 4; ++i) {
      for (int j = 0; j < 4; ++j) {
        res.m[i][j] = m[j][i];
      }
    }
    return res;
  }

  // 辅助函数：计算3x3矩阵的行列式
  float determinant3x3(float a, float b, float c, float d, float e, float f,
                       float g, float h, float i) const {
    return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
  }

  // 辅助函数：计算4x4矩阵的代数余子式
  float cofactor(int r, int c) const {
    // 创建一个3x3的子矩阵
    float sub[3][3];
    int sub_r = 0, sub_c = 0;
    for (int i = 0; i < 4; ++i) {
      if (i == r)
        continue;
      sub_c = 0;
      for (int j = 0; j < 4; ++j) {
        if (j == c)
          continue;
        sub[sub_r][sub_c] = m[i][j];
        sub_c++;
      }
      sub_r++;
    }
    float det =
        determinant3x3(sub[0][0], sub[0][1], sub[0][2], sub[1][0], sub[1][1],
                       sub[1][2], sub[2][0], sub[2][1], sub[2][2]);
    // 根据行列式的位置确定正负号
    return ((r + c) % 2 == 0 ? 1.0f : -1.0f) * det;
  }

  // 计算逆矩阵
  Mat4 inverse() const {
    Mat4 adj; // 伴随矩阵
    for (int i = 0; i < 4; ++i) {
      for (int j = 0; j < 4; ++j) {
        adj.m[j][i] = cofactor(i, j); // 注意：伴随矩阵是余子矩阵的转置
      }
    }

    float det = m[0][0] * adj.m[0][0] + m[0][1] * adj.m[1][0] +
                m[0][2] * adj.m[2][0] + m[0][3] * adj.m[3][0];

    if (std::abs(det) < std::numeric_limits<float>::epsilon()) {
      // 矩阵不可逆，返回单位矩阵或抛出错误
      return Mat4::identity();
    }

    Mat4 inv;
    float inv_det = 1.0f / det;
    for (int i = 0; i < 4; ++i) {
      for (int j = 0; j < 4; ++j) {
        inv.m[i][j] = adj.m[i][j] * inv_det;
      }
    }
    return inv;
  }

  // 创建透视投影矩阵
  static Mat4 perspective(float fov_deg, float aspect, float near, float far) {
    Mat4 res = Mat4::identity();
    float fov_rad = fov_deg * (std::numbers::pi / 180.0f);
    float tan_half_fov = std::tan(fov_rad / 2.0f);
    res.m[0][0] = 1.0f / (aspect * tan_half_fov);
    res.m[1][1] = 1.0f / tan_half_fov;
    res.m[2][2] = -(far + near) / (far - near);
    res.m[2][3] = -(2.0f * far * near) / (far - near);
    res.m[3][2] = -1.0f;
    return res;
  }

  // 创建正交投影矩阵
  static Mat4 orthographic(float left, float right, float bottom, float top,
                           float near, float far) {
    Mat4 res = Mat4::identity();
    res.m[0][0] = 2.0f / (right - left);
    res.m[1][1] = 2.0f / (top - bottom);
    res.m[2][2] = -2.0f / (far - near);
    res.m[0][3] = -(right + left) / (right - left);
    res.m[1][3] = -(top + bottom) / (top - bottom);
    res.m[2][3] = -(far + near) / (far - near);
    return res;
  }

  // 创建视图矩阵 (lookAt)
  static Mat4 lookAt(const Vec3 &eye, const Vec3 &target, const Vec3 &up) {
    Vec3 f = (eye - target).normalize(); // 摄像机Z轴 (从目标指向眼睛)
    Vec3 s = up.cross(f).normalize();    // 摄像机X轴
    Vec3 u = f.cross(s).normalize();     // 摄像机Y轴

    Mat4 res = Mat4::identity();

    // 旋转部分 (转置的基向量成为行)
    res.m[0][0] = s.x;
    res.m[0][1] = s.y;
    res.m[0][2] = s.z;
    res.m[1][0] = u.x;
    res.m[1][1] = u.y;
    res.m[1][2] = u.z;
    res.m[2][0] = f.x;
    res.m[2][1] = f.y;
    res.m[2][2] = f.z;
    res.m[0][3] = -s.dot(eye);
    res.m[1][3] = -u.dot(eye);
    res.m[2][3] = -f.dot(eye);

    return res;
  }

  // 创建平移矩阵
  static Mat4 translate(const Vec3 &v) {
    Mat4 res = Mat4::identity();
    res.m[0][3] = v.x;
    res.m[1][3] = v.y;
    res.m[2][3] = v.z;
    return res;
  }

  // 创建缩放矩阵
  static Mat4 scale(const Vec3 &v) {
    Mat4 res = Mat4::identity();
    res.m[0][0] = v.x;
    res.m[1][1] = v.y;
    res.m[2][2] = v.z;
    return res;
  }

  // 创建绕X轴旋转矩阵
  static Mat4 rotateX(float angle) {
    Mat4 res = Mat4::identity();
    float c = std::cos(angle);
    float s = std::sin(angle);
    res.m[1][1] = c;
    res.m[1][2] = -s;
    res.m[2][1] = s;
    res.m[2][2] = c;
    return res;
  }

  // 创建绕Y轴旋转矩阵
  static Mat4 rotateY(float angle) {
    Mat4 res = Mat4::identity();
    float c = std::cos(angle);
    float s = std::sin(angle);
    res.m[0][0] = c;
    res.m[0][2] = s;
    res.m[2][0] = -s;
    res.m[2][2] = c;
    return res;
  }

  // 创建绕Z轴旋转矩阵
  static Mat4 rotateZ(float angle) {
    Mat4 res = Mat4::identity();
    float c = std::cos(angle);
    float s = std::sin(angle);
    res.m[0][0] = c;
    res.m[0][1] = -s;
    res.m[1][0] = s;
    res.m[1][1] = c;
    return res;
  }
};

// ========================
// 工具函数
// ========================

// 将归一化设备坐标 (NDC) 转换为屏幕坐标
Vec3 ndcToScreen(Vec4 ndc, int W, int H) {
  // NDC的X, Y在[-1, 1]范围内。屏幕X, Y在[0, W]和[0, H]
  // 屏幕Y通常是翻转的 (左上角是0,0)
  return {(ndc.x * 0.5f + 0.5f) * W, (1.0f - (ndc.y * 0.5f + 0.5f)) * H, ndc.z};
}

// 2D三角形光栅化的边函数
float edgeFunc(Vec3 a, Vec3 b, Vec3 c) {
  return (c.x - a.x) * (b.y - a.y) - (c.y - a.y) * (b.x - a.x);
}

float lerp(float a, float b, float t) { return a + (b - a) * t; }

// 颜色结构，用于PPM输出
struct Color {
  unsigned char r, g, b;
};

// 顶点结构，包含位置、法线、切线和UV坐标
struct Vertex {
  Vec3 pos;     // 世界空间位置
  Vec3 normal;  // 世界空间法线
  Vec3 tangent; // 切线
  Vec2 uv;      // 纹理坐标
};

// 三角形结构
struct Triangle {
  Vertex v[3];
};

// ========================
// 纹理
// ========================

struct Texture {
  int w, h;
  std::vector<Color> data;

  // 生成一个简单的棋盘格纹理
  void generateChecker(int size) {
    w = h = size;
    data.resize(w * h);
    for (int y = 0; y < h; ++y) {
      for (int x = 0; x < w; ++x) {
        // 交替的黑白 (或深灰) 方块
        data[y * w + x] = ((x / 8) % 2 == (y / 8) % 2) ? Color{255, 255, 255}
                                                       : Color{50, 50, 50};
      }
    }
  }

  // 双线性纹理采样
  Color bilinear(Vec2 uv) const {
    // 将UV坐标包裹到[0, 1)范围
    uv.u -= std::floor(uv.u);
    uv.v -= std::floor(uv.v);

    // 将UV转换为像素坐标，偏移0.5f以指向像素中心
    float x = uv.u * w - 0.5f;
    float y = uv.v * h - 0.5f;

    // 获取周围四个像素的整数坐标
    int x0 = static_cast<int>(std::floor(x));
    int y0 = static_cast<int>(std::floor(y));
    int x1 = x0 + 1;
    int y1 = y0 + 1;

    // 插值的小数部分
    float u_frac = x - x0;
    float v_frac = y - y0;

    // 辅助函数：获取纹理数据中的颜色，并处理包裹
    auto get_color_wrapped = [&](int xx, int yy) {
      xx = (xx % w + w) % w; // 确保负值取模后为正
      yy = (yy % h + h) % h;
      return data[yy * w + xx];
    };

    // 辅助函数：颜色线性插值
    auto lerp_color = [](Color a, Color b, float t) {
      return Color{static_cast<unsigned char>(a.r * (1.0f - t) + b.r * t),
                   static_cast<unsigned char>(a.g * (1.0f - t) + b.g * t),
                   static_cast<unsigned char>(a.b * (1.0f - t) + b.b * t)};
    };

    // 双线性插值
    Color c00 = get_color_wrapped(x0, y0);
    Color c10 = get_color_wrapped(x1, y0);
    Color c01 = get_color_wrapped(x0, y1);
    Color c11 = get_color_wrapped(x1, y1);

    Color c_top = lerp_color(c00, c10, u_frac);
    Color c_bottom = lerp_color(c01, c11, u_frac);

    return lerp_color(c_top, c_bottom, v_frac);
  }
};

// ========== Shader 类型定义 ==========
enum class ShaderType {
  BlinnPhong,
  Normal,
  Texture,
  Bump,
  Displacement,
  Parallax, // 新增视差贴图shader类型
  Shadow,
  AO, // 环境光遮蔽
  AlphaBlend,
  SSAO,       // 屏幕空间环境光遮蔽
  SpotLight,  // 聚光灯
  Depth,      // 深度图输出
  LightDepth, // 光源深度图 (用于概念性Shadow Map生成)
  MultiLight, // 多光源着色器 (无真实阴影)
  NoMSAA,
  Bloom // 新增 Bloom 着色器
};

// 通用Shader输入输出接口

// 通用Shader输入输出接口
struct IShaderIO {
  virtual ~IShaderIO() {}
};

// BlinnPhong 着色器输入输出结构体
struct BlinnPhongShaderIO : public IShaderIO {
  Vec3 position_world;
  Vec3 normal_world;
  Vec3 tangent_world;
  Vec2 uv;
  float ndc_z = 0.0f;
  Vec4 position_clip;
  Vec3 color;
};

// Normal Shader 输入输出
struct NormalShaderIO : public IShaderIO {
  Vec3 position_world;
  Vec3 normal_world;
  Vec4 position_clip;
  Vec3 color;
};

// Texture Shader 输入输出
struct TextureShaderIO : public IShaderIO {
  Vec3 position_world;
  Vec2 uv;
  Vec4 position_clip;
  Vec3 color;
};

// Bump Shader 输入输出
struct BumpShaderIO : public IShaderIO {
  Vec3 position_world;
  Vec3 normal_world;
  Vec3 tangent_world;
  Vec2 uv;
  Vec4 position_clip;
  Vec3 color;
};

// Displacement Shader 输入输出
struct DisplacementShaderIO : public IShaderIO {
  Vec3 position_world;
  Vec3 normal_world;
  Vec3 tangent_world;
  Vec2 uv;
  Vec4 position_clip;
  Vec3 color;
};

// Shadow Shader 输入输出
struct ShadowShaderIO : public IShaderIO {
  Vec3 position_world;
  Vec3 normal_world;
  Vec4 position_clip;
  Vec3 color;
};

// AO Shader 输入输出
struct AOShaderIO : public IShaderIO {
  Vec3 position_world;
  Vec3 normal_world;
  Vec4 position_clip;
  Vec3 color;
};

// Alpha Blend Shader 输入输出
struct AlphaBlendShaderIO : public IShaderIO {
  Vec3 position_world;
  Vec2 uv;
  Vec4 position_clip;
  Vec3 color;
};

// SSAO Shader 输入输出
struct SSAOShaderIO : public IShaderIO {
  Vec3 position_world;
  Vec3 normal_world;
  Vec2 uv_screen_normalized;
  Vec4 position_clip;
  Vec3 color;
};

// SpotLight Shader 输入输出
struct SpotLightShaderIO : public IShaderIO {
  Vec3 position_world;
  Vec3 normal_world;
  Vec2 uv;
  Vec4 position_clip;
  Vec3 color;
};

// Depth Shader 输入输出
struct DepthShaderIO : public IShaderIO {
  Vec3 position_world;
  float ndc_z = 0.0f;
  Vec4 position_clip;
  Vec3 color;
};

// LightDepth Shader 输入输出
struct LightDepthShaderIO : public IShaderIO {
  Vec3 position_world;
  float ndc_z = 0.0f;
  Vec4 position_clip;
  Vec3 color;
};

// MultiLight Shader 输入输出
struct MultiLightShaderIO : public IShaderIO {
  Vec3 position_world;
  Vec3 normal_world;
  Vec2 uv;
  Vec4 position_clip;
  Vec3 color;
};

// Parallax Shader 输入输出
struct ParallaxShaderIO : public IShaderIO {
  Vec3 position_world;
  Vec3 normal_world;
  Vec3 tangent_world;
  Vec2 uv;
  Vec4 position_clip;
  Vec3 color;
};

struct BloomShaderIO : public IShaderIO {
  Vec3 position_world;
  Vec2 uv;
  Vec4 position_clip;
  Vec3 color;
  Vec3 bloom_color; // 新增 bloom_color 用于存储 Bloom 结果
};

// ========================
// Shader 解耦：接口与各实现
// ========================
struct IShader {
  // 顶点着色器：输入输出均为IShaderIO
  virtual void vertex(IShaderIO &io) = 0;
  // 片元着色器：输入输出均为IShaderIO
  virtual void fragment(IShaderIO &io) = 0;
  virtual ~IShader() {}
};

// Blinn-Phong Shader
struct BlinnPhongShader : public IShader {
  Mat4 MVP;             // 模型-视图-投影矩阵
  Mat4 NormalMatrix;    // 法线变换矩阵
  Vec3 eye_pos_world;   // 摄像机世界坐标
  Vec3 light_dir_world; // 世界空间光照方向 (指向光源)
  Texture diffuse_tex;  // 漫反射纹理
  float ambient_strength, diffuse_strength, specular_strength,
      shininess; // 材质属性

  BlinnPhongShader(const Mat4 &mvp, const Mat4 &nm, const Vec3 &eye,
                   const Vec3 &light_dir, const Texture &tex)
      : MVP(mvp), NormalMatrix(nm), eye_pos_world(eye),
        light_dir_world(light_dir), diffuse_tex(tex), ambient_strength(0.1f),
        diffuse_strength(0.7f), specular_strength(0.2f), shininess(32.0f) {}

  void vertex(IShaderIO &io) override {
    auto &data = static_cast<BlinnPhongShaderIO &>(io);
    data.position_clip =
        MVP.multiply({data.position_world.x, data.position_world.y,
                      data.position_world.z, 1.0f});
  }

  void fragment(IShaderIO &io) override {
    auto &data = static_cast<BlinnPhongShaderIO &>(io);
    Vec3 N = NormalMatrix
                 .multiply(Vec4(data.normal_world.x, data.normal_world.y,
                                data.normal_world.z, 0.0f))
                 .normalize();
    Vec3 L = (light_dir_world * -1.0f).normalize();
    Vec3 V = (eye_pos_world - data.position_world).normalize();
    Vec3 H = (L + V).normalize();

    float NdotL = std::max(0.0f, N.dot(L));
    float NdotH = std::max(0.0f, N.dot(H));
    float spec = std::pow(NdotH, shininess);

    Vec3 base_color = {1.0f, 1.0f, 1.0f};
    if (diffuse_tex.w > 0) {
      Color tex_color = diffuse_tex.bilinear(data.uv);
      base_color = {tex_color.r / 255.0f, tex_color.g / 255.0f,
                    tex_color.b / 255.0f};
    }

    Vec3 ambient_color = base_color * ambient_strength;
    Vec3 diffuse_color = base_color * diffuse_strength * NdotL;
    Vec3 specular_color = Vec3{1, 1, 1} * specular_strength * spec;

    Vec3 final_color = ambient_color + diffuse_color + specular_color;
    final_color.x = std::pow(final_color.x, 1.0f / 2.2f);
    final_color.y = std::pow(final_color.y, 1.0f / 2.2f);
    final_color.z = std::pow(final_color.z, 1.0f / 2.2f);

    data.color = final_color;
  }
};

// Normal Shader (可视化法线)
struct NormalShader : public IShader {
  Mat4 MVP;
  Mat4 NormalMatrix; // 法线变换矩阵
  NormalShader(const Mat4 &mvp, Mat4 &nm) : MVP(mvp), NormalMatrix(nm) {}

  void vertex(IShaderIO &io) override {
    auto &data = static_cast<NormalShaderIO &>(io);
    data.position_clip =
        MVP.multiply({data.position_world.x, data.position_world.y,
                      data.position_world.z, 1.0f});
  }

  void fragment(IShaderIO &io) override {
    auto &data = static_cast<NormalShaderIO &>(io);
    Vec3 n = NormalMatrix
                 .multiply(Vec4(data.normal_world.x, data.normal_world.y,
                                data.normal_world.z, 0.0f))
                 .normalize();
    data.color = {n.x * 0.5f + 0.5f, n.y * 0.5f + 0.5f, n.z * 0.5f + 0.5f};
  }
};

// Texture Shader (只显示纹理)
struct TextureShader : public IShader {
  Mat4 MVP;
  Texture diffuse_tex;
  Mat4 NormalMatrix;
  TextureShader(const Mat4 &mvp, const Mat4 &nm, const Texture &tex)
      : MVP(mvp), NormalMatrix(nm), diffuse_tex(tex) {}

  void vertex(IShaderIO &io) override {
    auto &data = static_cast<TextureShaderIO &>(io);
    data.position_clip =
        MVP.multiply({data.position_world.x, data.position_world.y,
                      data.position_world.z, 1.0f});
  }

  void fragment(IShaderIO &io) override {
    auto &data = static_cast<TextureShaderIO &>(io);
    Color tex_color = diffuse_tex.bilinear(data.uv);
    data.color = {tex_color.r / 255.0f, tex_color.g / 255.0f,
                  tex_color.b / 255.0f};
  }
};

// Bump Shader (法线贴图)
struct BumpShader : public IShader {
  Mat4 MVP;
  Vec3 eye_pos_world, light_dir_world;
  Texture diffuse_tex, normal_map_tex;
  float ambient_strength, diffuse_strength, specular_strength, shininess;
  Mat4 NormalMatrix;
  BumpShader(const Mat4 &mvp, const Mat4 &nm, const Vec3 &eye,
             const Vec3 &light_dir, const Texture &tex,
             const Texture &normalmap)
      : MVP(mvp), NormalMatrix(nm), eye_pos_world(eye),
        light_dir_world(light_dir), diffuse_tex(tex), normal_map_tex(normalmap),
        ambient_strength(0.1f), diffuse_strength(0.7f), specular_strength(0.2f),
        shininess(32.0f) {}

  void vertex(IShaderIO &io) override {
    auto &data = static_cast<BumpShaderIO &>(io);
    data.position_clip =
        MVP.multiply({data.position_world.x, data.position_world.y,
                      data.position_world.z, 1.0f});
  }

  void fragment(IShaderIO &io) override {
    auto &data = static_cast<BumpShaderIO &>(io);
    Vec3 N = NormalMatrix
                 .multiply(Vec4(data.normal_world.x, data.normal_world.y,
                                data.normal_world.z, 0.0f))
                 .normalize();
    Vec3 T = data.tangent_world.normalize();
    T = (T - N * N.dot(T)).normalize();
    Vec3 B = N.cross(T);
    if (normal_map_tex.w > 0) {
      Color normal_map_sample = normal_map_tex.bilinear(data.uv);
      Vec3 normal_tangent_space = {normal_map_sample.r / 255.0f * 2.0f - 1.0f,
                                   normal_map_sample.g / 255.0f * 2.0f - 1.0f,
                                   normal_map_sample.b / 255.0f * 2.0f - 1.0f};
      normal_tangent_space = normal_tangent_space.normalize();
      Vec3 perturbed_normal_world;
      perturbed_normal_world.x = T.x * normal_tangent_space.x +
                                 B.x * normal_tangent_space.y +
                                 N.x * normal_tangent_space.z;
      perturbed_normal_world.y = T.y * normal_tangent_space.x +
                                 B.y * normal_tangent_space.y +
                                 N.y * normal_tangent_space.z;
      perturbed_normal_world.z = T.z * normal_tangent_space.x +
                                 B.z * normal_tangent_space.y +
                                 N.z * normal_tangent_space.z;
      N = perturbed_normal_world.normalize();
    }
    Vec3 L = (light_dir_world * -1.0f).normalize();
    Vec3 V = (eye_pos_world - data.position_world).normalize();
    Vec3 H = (L + V).normalize();
    float NdotL = std::max(0.0f, N.dot(L));
    float NdotH = std::max(0.0f, N.dot(H));
    float spec = std::pow(NdotH, shininess);
    Vec3 base_color = {1.0f, 1.0f, 1.0f};
    if (diffuse_tex.w > 0) {
      Color tex_color = diffuse_tex.bilinear(data.uv);
      base_color = {tex_color.r / 255.0f, tex_color.g / 255.0f,
                    tex_color.b / 255.0f};
    }
    Vec3 final_color = base_color * ambient_strength +
                       base_color * diffuse_strength * NdotL +
                       Vec3{1, 1, 1} * specular_strength * spec;
    final_color.x = std::pow(final_color.x, 1.0f / 2.2f);
    final_color.y = std::pow(final_color.y, 1.0f / 2.2f);
    final_color.z = std::pow(final_color.z, 1.0f / 2.2f);
    data.color = final_color;
  }
};

// Displacement Shader (位移贴图)
struct DisplacementShader : public IShader {
  Mat4 MVP;
  Vec3 eye_pos_world, light_dir_world;
  Texture diffuse_tex, displacement_map_tex;
  float ambient_strength, diffuse_strength, specular_strength, shininess;
  float displacement_scale; // 位移强度
  Mat4 NormalMatrix;
  DisplacementShader(const Mat4 &mvp, const Mat4 &nm, const Vec3 &eye,
                     const Vec3 &light_dir, const Texture &tex,
                     const Texture &dispmap, float scale = 0.1f)
      : MVP(mvp), NormalMatrix(nm), eye_pos_world(eye),
        light_dir_world(light_dir), diffuse_tex(tex),
        displacement_map_tex(dispmap), ambient_strength(0.1f),
        diffuse_strength(0.7f), specular_strength(0.2f), shininess(32.0f),
        displacement_scale(scale) {}

  void vertex(IShaderIO &io) override {
    auto &data = static_cast<DisplacementShaderIO &>(io);
    Vec3 pos = data.position_world;
    if (displacement_map_tex.w > 0) {
      float h = displacement_map_tex.bilinear(data.uv).r / 255.0f;
      pos = pos + data.normal_world * (h - 0.5f) * displacement_scale;
    }
    data.position_clip = MVP.multiply({pos.x, pos.y, pos.z, 1.0f});
  }

  void fragment(IShaderIO &io) override {
    auto &data = static_cast<DisplacementShaderIO &>(io);
    // 构造TBN矩阵
    Vec3 N = NormalMatrix
                 .multiply(Vec4(data.normal_world.x, data.normal_world.y,
                                data.normal_world.z, 0.0f))
                 .normalize();
    Vec3 T = data.tangent_world.normalize();
    T = (T - N * N.dot(T)).normalize();
    Vec3 B = N.cross(T);

    // 视线和光线转到切线空间
    Vec3 V = (eye_pos_world - data.position_world).normalize();
    Vec3 L = (light_dir_world * -1.0f).normalize();
    Vec3 V_tangent = Vec3{V.dot(T), V.dot(B), V.dot(N)};
    Vec3 L_tangent = Vec3{L.dot(T), L.dot(B), L.dot(N)};
    Vec3 H_tangent = (L_tangent + V_tangent).normalize();
    // 采样和着色
    float NdotL = std::max(0.0f, N.dot(L));
    float NdotH = std::max(0.0f, N.dot((L + V).normalize()));
    float spec = std::pow(NdotH, shininess);
    Vec3 base_color = {1.0f, 1.0f, 1.0f};
    if (diffuse_tex.w > 0) {
      Color tex_color = diffuse_tex.bilinear(data.uv);
      base_color = {tex_color.r / 255.0f, tex_color.g / 255.0f,
                    tex_color.b / 255.0f};
    }
    Vec3 final_color = base_color * ambient_strength +
                       base_color * diffuse_strength * NdotL +
                       Vec3{1, 1, 1} * specular_strength * spec;
    final_color.x = std::pow(final_color.x, 1.0f / 2.2f);
    final_color.y = std::pow(final_color.y, 1.0f / 2.2f);
    final_color.z = std::pow(final_color.z, 1.0f / 2.2f);
    data.color = final_color;
  }
};

// Parallax Shader (视差贴图)
struct ParallaxShader : public IShader {
  Mat4 MVP;
  Vec3 eye_pos_world, light_dir_world;
  Texture diffuse_tex, parallax_map_tex, normal_map_tex;
  float ambient_strength, diffuse_strength, specular_strength, shininess;
  float parallax_scale; // 视差强度
  Mat4 NormalMatrix;
  ParallaxShader(const Mat4 &mvp, const Mat4 &nm, const Vec3 &eye,
                 const Vec3 &light_dir, const Texture &tex,
                 const Texture &parallaxmap, const Texture &nor_map_tex,
                 float scale = 0.05f)
      : MVP(mvp), NormalMatrix(nm), eye_pos_world(eye),
        light_dir_world(light_dir), diffuse_tex(tex),
        parallax_map_tex(parallaxmap), normal_map_tex(nor_map_tex),
        ambient_strength(0.1f), diffuse_strength(0.7f), specular_strength(0.2f),
        shininess(32.0f), parallax_scale(scale) {}

  void vertex(IShaderIO &io) override {
    auto &data = static_cast<ParallaxShaderIO &>(io);
    // data.position_clip = MVP.multiply({data.position_world.x,
    // data.position_world.y, data.position_world.z, 1.0f});
    data.position_clip = {data.position_world.x, data.position_world.y,
                          data.position_world.z,
                          1.0f}; // bricks2直接使用世界坐标
  }

  void fragment(IShaderIO &io) override {
    auto &data = static_cast<ParallaxShaderIO &>(io);
    // 构造TBN矩阵
    // Vec3 N = NormalMatrix.multiply(Vec4(data.normal_world.x,
    // data.normal_world.y, data.normal_world.z, 0.0f)).normalize();
    Vec3 N = data.normal_world.normalize();
    Vec3 T = data.tangent_world.normalize();
    T = (T - N * N.dot(T)).normalize();
    Vec3 B = N.cross(T);
    // 视线转到切线空间
    Vec3 V = (eye_pos_world - data.position_world).normalize();
    Vec3 V_tangent = Vec3{V.dot(T), V.dot(B), V.dot(N)};
    // 视差贴图采样
    Vec2 uv = data.uv;
    if (parallax_map_tex.w > 0) {
      //   float h = parallax_map_tex.bilinear(uv).r / 255.0f;
      //   float parallax_offset = (h - 0.5f) * parallax_scale;
      //   float view_z = std::abs(V_tangent.z) + 1e-4f;
      //   uv = uv + Vec2{V_tangent.x, V_tangent.y} * (parallax_offset /
      //   view_z); uv.u = std::max(0.0f, std::min(1.0f, uv.u)); uv.v =
      //   std::max(0.0f, std::min(1.0f, uv.v));

      // number of depth layers
      const float minLayers = 8;
      const float maxLayers = 32;
      float numLayers =
          lerp(maxLayers, minLayers, abs(Vec3(0.0, 0.0, 1.0).dot(V_tangent)));
      // calculate the size of each layer
      float layerDepth = 1.0 / numLayers;
      // depth of current layer
      float currentLayerDepth = 0.0;
      // the amount to shift the texture coordinates per layer (from vector P)
      Vec2 P = Vec2{V_tangent.x, V_tangent.y} / V_tangent.z * parallax_scale;
      Vec2 deltaTexCoords = P / numLayers;

      // get initial values
      Vec2 currentTexCoords = uv;
      float currentDepthMapValue =
          parallax_map_tex.bilinear(currentTexCoords).r / 255.0f;

      while (currentLayerDepth < currentDepthMapValue) {
        // shift texture coordinates along direction of P
        currentTexCoords -= deltaTexCoords;
        // get depthmap value at current texture coordinates
        currentDepthMapValue =
            parallax_map_tex.bilinear(currentTexCoords).r / 255.0f;
        // get depth of next layer
        currentLayerDepth += layerDepth;
      }

      // get texture coordinates before collision (reverse operations)
      Vec2 prevTexCoords = currentTexCoords + deltaTexCoords;

      // get depth after and before collision for linear interpolation
      float afterDepth = currentDepthMapValue - currentLayerDepth;
      float beforeDepth =
          parallax_map_tex.bilinear(currentTexCoords).r / 255.0f -
          currentLayerDepth + layerDepth;

      // interpolation of texture coordinates
      float weight = afterDepth / (afterDepth - beforeDepth);
      Vec2 finalTexCoords =
          prevTexCoords * weight + currentTexCoords * (1.0 - weight);

      uv = finalTexCoords;

      if (uv.u < 0.0f || uv.u > 1.0f || uv.v < 0.0f || uv.v > 1.0f) {
        data.color = Vec3(0.0f, 0.0f, 0.0f); // 如果UV超出范围，返回黑色,discard
        return;
      }
      // uv.u = std::max(0.0f, std::min(1.0f, uv.u));
      // uv.v = std::max(0.0f, std::min(1.0f, uv.v));
    }

    Vec3 N_perturbed_tangent_space;
    if (normal_map_tex.w > 0) {
      Color normal_map_sample = normal_map_tex.bilinear(uv); // 使用偏移后的UV
      N_perturbed_tangent_space = {normal_map_sample.r / 255.0f * 2.0f - 1.0f,
                                   normal_map_sample.g / 255.0f * 2.0f - 1.0f,
                                   normal_map_sample.b / 255.0f * 2.0f - 1.0f};
      // N = N_perturbed_tangent_space.normalize();
    }

    // 光照方向也可转到切线空间用于更真实的高光
    Vec3 L = (light_dir_world * -1.0f).normalize();
    Vec3 L_tangent = Vec3{L.dot(T), L.dot(B), L.dot(N)};
    Vec3 H_tangent = (L_tangent + V_tangent).normalize();
    Vec3 base_color = {1.0f, 1.0f, 1.0f};
    if (diffuse_tex.w > 0) {
      Color tex_color = diffuse_tex.bilinear(uv);
      base_color = {tex_color.r / 255.0f, tex_color.g / 255.0f,
                    tex_color.b / 255.0f};
    }
    float NdotL = std::max(0.0f, N_perturbed_tangent_space.dot(L_tangent));
    float NdotH = std::max(0.0f, N_perturbed_tangent_space.dot(H_tangent));
    float spec = std::pow(NdotH, shininess);
    Vec3 final_color = base_color * ambient_strength +
                       base_color * diffuse_strength * NdotL +
                       Vec3{1, 1, 1} * specular_strength * spec;
    // final_color = Vec3{1.0,1.0,1.0};
    final_color.x = std::pow(final_color.x, 1.0f / 2.2f);
    final_color.y = std::pow(final_color.y, 1.0f / 2.2f);
    final_color.z = std::pow(final_color.z, 1.0f / 2.2f);
    data.color = final_color;
  }
};

// Shadow Shader (简化阴影可视化)
struct ShadowShader : public IShader {
  Mat4 MVP;
  Vec3 light_dir_world; // 光照方向
  Mat4 NormalMatrix;
  ShadowShader(const Mat4 &mvp, const Mat4 &nm, const Vec3 &light_dir)
      : MVP(mvp), NormalMatrix(nm), light_dir_world(light_dir) {}

  void vertex(IShaderIO &io) override {
    auto &data = static_cast<ShadowShaderIO &>(io);
    data.position_clip =
        MVP.multiply({data.position_world.x, data.position_world.y,
                      data.position_world.z, 1.0f});
  }

  void fragment(IShaderIO &io) override {
    auto &data = static_cast<ShadowShaderIO &>(io);
    Vec3 N = NormalMatrix
                 .multiply(Vec4(data.normal_world.x, data.normal_world.y,
                                data.normal_world.z, 0.0f))
                 .normalize();
    Vec3 L = (light_dir_world * -1.0f).normalize();
    float NdotL = std::max(0.0f, N.dot(L));
    float shadow_factor = NdotL < 0.3f ? 0.3f : 1.0f;
    data.color = Vec3{shadow_factor, shadow_factor, shadow_factor};
  }
};

// AO Shader (改进: 模拟方向性环境光遮蔽)
// 这是一个简化的环境光遮蔽，它根据法线方向相对于一个“环境光方向”来计算遮蔽。
// 更真实的AO需要考虑周围几何体。
struct AOShader : public IShader {
  Mat4 MVP;
  Vec3 ambient_light_dir;
  Mat4 NormalMatrix;
  AOShader(const Mat4 &mvp, const Mat4 &nm,
           const Vec3 &amb_dir = Vec3{0.0f, 1.0f, 0.0f})
      : MVP(mvp), NormalMatrix(nm), ambient_light_dir(amb_dir.normalize()) {}

  void vertex(IShaderIO &io) override {
    auto &data = static_cast<AOShaderIO &>(io);
    data.position_clip =
        MVP.multiply({data.position_world.x, data.position_world.y,
                      data.position_world.z, 1.0f});
  }

  void fragment(IShaderIO &io) override {
    auto &data = static_cast<AOShaderIO &>(io);
    Vec3 N = data.normal_world.normalize();
    float ao = std::max(0.0f, N.dot(ambient_light_dir));
    ao = std::pow(ao, 0.5f);
    data.color = Vec3{ao, ao, ao};
  }
};

// Alpha Blend Shader (透明混合)
struct AlphaBlendShader : public IShader {
  Mat4 MVP;
  Texture diffuse_tex;
  float alpha_value;
  Mat4 NormalMatrix;
  AlphaBlendShader(const Mat4 &mvp, const Mat4 &nm, const Texture &tex,
                   float alpha = 0.5f)
      : MVP(mvp), NormalMatrix(nm), diffuse_tex(tex), alpha_value(alpha) {}

  void vertex(IShaderIO &io) override {
    auto &data = static_cast<AlphaBlendShaderIO &>(io);
    data.position_clip =
        MVP.multiply({data.position_world.x, data.position_world.y,
                      data.position_world.z, 1.0f});
  }

  void fragment(IShaderIO &io) override {
    auto &data = static_cast<AlphaBlendShaderIO &>(io);
    Color tex_color = diffuse_tex.bilinear(data.uv);
    Vec3 base = {tex_color.r / 255.0f, tex_color.g / 255.0f,
                 tex_color.b / 255.0f};
    Vec3 bg = {1.0f, 1.0f, 1.0f};
    data.color = base * alpha_value + bg * (1.0f - alpha_value);
  }
};

// SSAO Shader (改进: 屏幕空间环境光遮蔽模拟)
// 这是一个简化的SSAO，模拟在屏幕空间基于深度缓冲的遮蔽。
// 它会根据像素周围深度值的变化来计算遮蔽，而不是实际的几何体遮挡。
// 需要访问深度缓冲区，但在这个软光栅实现中，我们只能模拟局部深度变化。
struct SSAOShader : public IShader {
  Mat4 MVP;
  Mat4 ViewMatrix;
  float near_plane_z, far_plane_z;
  std::vector<Vec3> ssao_samples;
  Texture noise_texture;
  Mat4 NormalMatrix;
  SSAOShader(const Mat4 &mvp, const Mat4 &nm, const Mat4 &view_mat,
             float near_z, float far_z, int num_samples = 16)
      : MVP(mvp), NormalMatrix(nm), ViewMatrix(view_mat), near_plane_z(near_z),
        far_plane_z(far_z) {
    std::default_random_engine generator;
    std::uniform_real_distribution<float> random_floats(0.0, 1.0);
    for (int i = 0; i < num_samples; ++i) {
      Vec3 sample(random_floats(generator) * 2.0f - 1.0f,
                  random_floats(generator) * 2.0f - 1.0f,
                  random_floats(generator));
      sample = sample.normalize();
      sample = sample * random_floats(generator);
      float scale = (float)i / (float)num_samples;
      scale = 0.1f + scale * scale * 0.9f;
      sample = sample * scale;
      ssao_samples.push_back(sample);
    }
    int noise_size = 4;
    noise_texture.w = noise_texture.h = noise_size;
    noise_texture.data.resize(noise_size * noise_size);
    for (int i = 0; i < noise_size * noise_size; ++i) {
      Vec3 noise(random_floats(generator) * 2.0f - 1.0f,
                 random_floats(generator) * 2.0f - 1.0f, 0.0f);
      noise = noise.normalize();
      noise_texture.data[i] =
          Color{static_cast<unsigned char>((noise.x * 0.5f + 0.5f) * 255),
                static_cast<unsigned char>((noise.y * 0.5f + 0.5f) * 255), 0};
    }
  }

  void vertex(IShaderIO &io) override {
    auto &data = static_cast<SSAOShaderIO &>(io);
    data.position_clip =
        MVP.multiply({data.position_world.x, data.position_world.y,
                      data.position_world.z, 1.0f});
  }

  void fragment(IShaderIO &io) override {
    auto &data = static_cast<SSAOShaderIO &>(io);
    Vec4 pos_view_4 =
        ViewMatrix.multiply({data.position_world.x, data.position_world.y,
                             data.position_world.z, 1.0f});
    Vec3 pos_view = Vec3(pos_view_4.x, pos_view_4.y, pos_view_4.z);
    Vec3 normal_view =
        NormalMatrix
            .multiply(Vec4(data.normal_world.x, data.normal_world.y,
                           data.normal_world.z, 0.0f))
            .normalize();
    Color noise_color = noise_texture.bilinear(data.uv_screen_normalized *
                                               (noise_texture.w / 4.0f));
    Vec3 random_vec = {noise_color.r / 255.0f * 2.0f - 1.0f,
                       noise_color.g / 255.0f * 2.0f - 1.0f, 0.0f};
    random_vec = random_vec.normalize();
    Vec3 tangent = random_vec.cross(normal_view);
    if (tangent.length() < 0.001f) {
      tangent = Vec3(1.0f, 0.0f, 0.0f).cross(normal_view).normalize();
    } else {
      tangent = tangent.normalize();
    }
    Vec3 bitangent = normal_view.cross(tangent);
    float occlusion = 0.0f;
    int kernel_size = ssao_samples.size();
    float radius = 0.5f;
    for (int i = 0; i < kernel_size; ++i) {
      Vec3 sample_offset = ssao_samples[i];
      Vec3 rotated_sample = tangent * sample_offset.x +
                            bitangent * sample_offset.y +
                            normal_view * sample_offset.z;
      Vec3 sample_pos_view = pos_view + rotated_sample * radius;
      float sample_dot_normal = sample_offset.dot(normal_view);
      float dist_from_center = sample_offset.length();
      if (sample_dot_normal < 0.0f) {
        float strength = std::max(0.0f, 1.0f - dist_from_center / radius);
        occlusion += strength;
      }
    }
    occlusion /= (float)kernel_size;
    occlusion = 1.0f - occlusion;
    occlusion = std::pow(occlusion, 1.0f);
    occlusion = std::max(0.0f, std::min(1.0f, occlusion));
    data.color = Vec3{occlusion, occlusion, occlusion};
  }
};

// SpotLight Shader (聚光灯)
struct SpotLightShader : public IShader {
  Mat4 MVP;
  Vec3 eye_pos_world;                       // 摄像机世界坐标
  Vec3 light_pos_world;                     // 光源世界坐标
  Vec3 spotlight_dir_world;                 // 聚光灯方向 (从光源发出)
  Texture diffuse_tex;                      // 漫反射纹理
  float inner_cone_angle, outer_cone_angle; // 内锥角和外锥角 (度)
  float ambient_strength, diffuse_strength, specular_strength, shininess;

  SpotLightShader(const Mat4 &mvp, const Vec3 &eye, const Vec3 &light_pos,
                  const Vec3 &spot_dir, const Texture &tex, float inner_angle,
                  float outer_angle)
      : MVP(mvp), eye_pos_world(eye), light_pos_world(light_pos),
        spotlight_dir_world(spot_dir), diffuse_tex(tex),
        inner_cone_angle(inner_angle), outer_cone_angle(outer_angle),
        ambient_strength(0.1f), diffuse_strength(0.7f), specular_strength(0.2f),
        shininess(32.0f) {}

  void vertex(IShaderIO &io) override {
    auto &data = static_cast<SpotLightShaderIO &>(io);
    data.position_clip =
        MVP.multiply({data.position_world.x, data.position_world.y,
                      data.position_world.z, 1.0f});
  }

  void fragment(IShaderIO &io) override {
    auto &data = static_cast<SpotLightShaderIO &>(io);
    Vec3 light_color = {1.0f, 1.0f, 1.0f};
    Vec3 N = data.normal_world.normalize();
    Vec3 L = (light_pos_world - data.position_world).normalize();
    Vec3 V = (eye_pos_world - data.position_world).normalize();
    Vec3 H = (L + V).normalize();
    Vec3 spotDir = spotlight_dir_world.normalize();
    float spotCos = L.dot(-spotDir);
    float innerCos = std::cos(inner_cone_angle * std::numbers::pi / 180.0f);
    float outerCos = std::cos(outer_cone_angle * std::numbers::pi / 180.0f);
    float intensity = 0.0f;
    if (spotCos > outerCos) {
      if (spotCos >= innerCos) {
        intensity = 1.0f;
      } else {
        float t = (spotCos - outerCos) / (innerCos - outerCos);
        intensity = t;
      }
    } else {
      intensity = 0.0f;
    }
    float NdotL = std::max(0.0f, N.dot(L));
    float NdotH = std::max(0.0f, N.dot(H));
    float spec = std::pow(NdotH, shininess);
    Vec3 base_color = {1.0f, 1.0f, 1.0f};
    if (diffuse_tex.w > 0) {
      Color tex_color = diffuse_tex.bilinear(data.uv);
      base_color = {tex_color.r / 255.0f, tex_color.g / 255.0f,
                    tex_color.b / 255.0f};
    }
    Vec3 final_color = base_color * ambient_strength +
                       light_color * diffuse_strength * NdotL * intensity +
                       light_color * specular_strength * spec * intensity;
    final_color.x = std::pow(final_color.x, 1.0f / 2.2f);
    final_color.y = std::pow(final_color.y, 1.0f / 2.2f);
    final_color.z = std::pow(final_color.z, 1.0f / 2.2f);
    data.color = final_color;
  }
};

// Depth Shader (深度图输出)
// 将NDC空间下的深度值映射到灰度颜色
struct DepthShader : public IShader {
  Mat4 MVP;

  DepthShader(const Mat4 &mvp) : MVP(mvp) {}

  void vertex(IShaderIO &io) override {
    auto &data = static_cast<DepthShaderIO &>(io);
    data.position_clip =
        MVP.multiply({data.position_world.x, data.position_world.y,
                      data.position_world.z, 1.0f});
  }

  void fragment(IShaderIO &io) override {
    auto &data = static_cast<DepthShaderIO &>(io);
    float depth_0_1 = (data.ndc_z + 1.0f) * 0.5f;
    float exponent = 3.0f;
    float visual_depth = std::pow(depth_0_1, exponent);
    visual_depth = 1.0f - visual_depth;
    visual_depth = std::max(0.0f, std::min(1.0f, visual_depth));
    data.color = Vec3{visual_depth, visual_depth, visual_depth};
  }
};

// LightDepth Shader (光源深度图，用于概念性Shadow Map生成)
// 这个shader输出从光源视角看到的深度，理论上用于生成Shadow Map
struct LightDepthShader : public IShader {
  Mat4 LightMVP; // 光源的Model-View-Projection矩阵

  LightDepthShader(const Mat4 &light_mvp) : LightMVP(light_mvp) {}

  void vertex(IShaderIO &io) override {
    auto &data = static_cast<LightDepthShaderIO &>(io);
    data.position_clip =
        LightMVP.multiply({data.position_world.x, data.position_world.y,
                           data.position_world.z, 1.0f});
  }

  void fragment(IShaderIO &io) override {
    auto &data = static_cast<LightDepthShaderIO &>(io);
    float depth_normalized = (data.ndc_z + 1.0f) * 0.5f;
    data.color = Vec3{depth_normalized, depth_normalized, depth_normalized};
  }
};

// 光源结构体
struct Light {
  Vec3 position;   // 光源位置 (用于点光源)
  Vec3 direction;  // 光源方向 (从物体指向光源，用于平行光和聚光灯)
  Vec3 color;      // 光源颜色
  float intensity; // 光源强度
};

// MultiLight Blinn-Phong Shader (多光源，无真实阴影)
struct MultiLightBlinnPhongShader : public IShader {
  Mat4 MVP;
  Vec3 eye_pos_world;
  std::vector<Light> lights; // 光源列表
  Texture diffuse_tex;
  float ambient_strength, diffuse_strength, specular_strength, shininess;

  MultiLightBlinnPhongShader(const Mat4 &mvp, const Vec3 &eye,
                             const std::vector<Light> &light_list,
                             const Texture &tex)
      : MVP(mvp), eye_pos_world(eye), lights(light_list), diffuse_tex(tex),
        ambient_strength(0.1f), diffuse_strength(0.7f), specular_strength(0.2f),
        shininess(32.0f) {}
  void vertex(IShaderIO &io) override {
    auto &data = static_cast<MultiLightShaderIO &>(io);
    data.position_clip =
        MVP.multiply({data.position_world.x, data.position_world.y,
                      data.position_world.z, 1.0f});
  }

  void fragment(IShaderIO &io) override {
    auto &data = static_cast<MultiLightShaderIO &>(io);
    Vec3 N = data.normal_world.normalize();
    Vec3 V = (eye_pos_world - data.position_world).normalize();
    Vec3 final_color_accum = {0.0f, 0.0f, 0.0f};
    Vec3 base_color = {1.0f, 1.0f, 1.0f};
    if (diffuse_tex.w > 0) {
      Color tex_color = diffuse_tex.bilinear(data.uv);
      base_color = {tex_color.r / 255.0f, tex_color.g / 255.0f,
                    tex_color.b / 255.0f};
    }
    final_color_accum = final_color_accum + base_color * ambient_strength;
    for (const auto &light : lights) {
      Vec3 L = (light.position - data.position_world).normalize();
      Vec3 H = (L + V).normalize();
      float NdotL = std::max(0.0f, N.dot(L));
      float NdotH = std::max(0.0f, N.dot(H));
      float spec = std::pow(NdotH, shininess);
      Vec3 diffuse_term =
          base_color * diffuse_strength * NdotL * light.color * light.intensity;
      Vec3 specular_term = Vec3{1, 1, 1} * specular_strength * spec *
                           light.color * light.intensity;
      final_color_accum = final_color_accum + diffuse_term + specular_term;
    }
    final_color_accum.x = std::pow(final_color_accum.x, 1.0f / 2.2f);
    final_color_accum.y = std::pow(final_color_accum.y, 1.0f / 2.2f);
    final_color_accum.z = std::pow(final_color_accum.z, 1.0f / 2.2f);
    data.color = final_color_accum;
  }
};

struct BloomShader: public IShader {
  Mat4 MVP;
  Texture input_texture; // 输入纹理
  float bloom_strength;  // Bloom强度

  BloomShader(const Mat4 &mvp, const Texture &tex, float strength = 1.0f)
      : MVP(mvp), input_texture(tex), bloom_strength(strength) {}

  void vertex(IShaderIO &io) override {
    auto &data = static_cast<BloomShaderIO &>(io);
    data.position_clip =
        MVP.multiply({data.position_world.x, data.position_world.y,
                      data.position_world.z, 1.0f});
  }

  void fragment(IShaderIO &io) override {
    auto &data = static_cast<BloomShaderIO &>(io);
    Color tex_color = input_texture.bilinear(data.uv);
    Vec3 color = {tex_color.r / 255.0f, tex_color.g / 255.0f, tex_color.b / 255.0f};
    float lum = 0.2126f * color.x + 0.7152f * color.y + 0.0722f * color.z;
    if (lum > 0.7f) { // 阈值可调
      data.bloom_color = color * bloom_strength;
    } else {
      data.bloom_color = Vec3{0.0f, 0.0f, 0.0f};
    }
  }
};

// ========================
// MSAA (Multi-Sample Anti-Aliasing)
// ========================
const int MSAA_SAMPLES_1 = 1;
const float MSAA_OFFSETS_1[1][2] = {{0.5f, 0.5f}};
const int MSAA_SAMPLES_4 = 4;
const float MSAA_OFFSETS_4[4][2] = {
    {0.375f, 0.125f}, {0.875f, 0.375f}, {0.125f, 0.625f}, {0.625f, 0.875f}};
// 默认MSAA设置为4x
int MSAA_SAMPLES = MSAA_SAMPLES_4;
const float (*MSAA_OFFSETS)[2] = MSAA_OFFSETS_4;

// ========================
// 模型加载函数（只支持三角面）
// ========================
struct Mesh {
  std::vector<Triangle> triangles;
};

// 从.obj文件加载模型
Mesh loadModel(const std::string &filename) {
  Mesh mesh;
  tinyobj::attrib_t attrib;
  std::vector<tinyobj::shape_t> shapes;
  std::vector<tinyobj::material_t> materials;
  std::string warn, err;

  // 使用tinyobjloader加载OBJ
  if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err,
                        filename.c_str(), nullptr)) {
    std::cerr << "Failed to load obj: " << filename << std::endl;
    if (!warn.empty())
      std::cerr << "WARN: " << warn << std::endl;
    if (!err.empty())
      std::cerr << "ERR: " << err << std::endl;
    return mesh;
  }

  // 遍历所有形状和面，构建三角形列表
  for (const auto &shape : shapes) {
    const auto &indices = shape.mesh.indices;
    // tinyobjloader将所有面展平为三角形索引，所以可以直接每三个索引构成一个三角形
    for (size_t f = 0; f + 2 < indices.size(); f += 3) {
      Triangle tri;
      for (int v_idx_in_tri = 0; v_idx_in_tri < 3; ++v_idx_in_tri) {
        const tinyobj::index_t &idx = indices[f + v_idx_in_tri];

        // 顶点坐标
        tri.v[v_idx_in_tri].pos.x = attrib.vertices[3 * idx.vertex_index + 0];
        tri.v[v_idx_in_tri].pos.y = attrib.vertices[3 * idx.vertex_index + 1];
        tri.v[v_idx_in_tri].pos.z = attrib.vertices[3 * idx.vertex_index + 2];

        // 法线
        if (!attrib.normals.empty() && idx.normal_index >= 0) {
          tri.v[v_idx_in_tri].normal.x =
              attrib.normals[3 * idx.normal_index + 0];
          tri.v[v_idx_in_tri].normal.y =
              attrib.normals[3 * idx.normal_index + 1];
          tri.v[v_idx_in_tri].normal.z =
              attrib.normals[3 * idx.normal_index + 2];
        } else {
          // 如果模型没有提供法线，则给一个默认值
          tri.v[v_idx_in_tri].normal = {0, 0, 1};
        }

        // 纹理坐标
        if (!attrib.texcoords.empty() && idx.texcoord_index >= 0) {
          tri.v[v_idx_in_tri].uv.u =
              attrib.texcoords[2 * idx.texcoord_index + 0];
          tri.v[v_idx_in_tri].uv.v =
              attrib.texcoords[2 * idx.texcoord_index + 1];
        } else {
          // 如果模型没有提供UV，则给一个默认值
          tri.v[v_idx_in_tri].uv = {0, 0};
        }
        // 切线 (这里只是一个占位符，实际需要根据UV和位置计算)
        // 在加载模型时计算切线是一个复杂的过程，通常需要处理UV缝合、法线翻转等问题。
        // 简化起见，这里直接给一个默认值，但在bump shader中会根据法线进行调整。
        tri.v[v_idx_in_tri].tangent = {1, 0, 0};
      }
      mesh.triangles.push_back(tri);
    }
  }
  return mesh;
}

// ========================
// 纹理加载函数
// ========================
Texture loadTexture(const std::string &filename) {
  Texture tex;
  int n; // 颜色通道数
  stbi_set_flip_vertically_on_load(
      true); // 图像通常Y轴向上，但OpenGL和大多数UV习惯Y轴向下
  unsigned char *data =
      stbi_load(filename.c_str(), &tex.w, &tex.h, &n, 3); // 强制加载3通道RGB
  if (!data) {
    std::cerr << "Failed to load texture: " << filename << std::endl;
    return tex;
  }
  tex.data.resize(tex.w * tex.h);
  for (int i = 0; i < tex.w * tex.h; ++i) {
    tex.data[i].r = data[3 * i + 0];
    tex.data[i].g = data[3 * i + 1];
    tex.data[i].b = data[3 * i + 2];
  }
  stbi_image_free(data); // 释放stb_image加载的内存
  return tex;
}

// ========================
// 重心坐标插值函数 (模板化以支持多种类型)
// ========================
template <typename T>
T baryInterp(const T &v0, const T &v1, const T &v2, float w0, float w1,
             float w2) {
  return v0 * w0 + v1 * w1 + v2 * w2;
}

// 获取MSAA采样点在像素内的偏移
Vec3 getSamplePoint(int x, int y, int s, const float MSAA_OFFSETS[][2]) {
  return {x + MSAA_OFFSETS[s][0], y + MSAA_OFFSETS[s][1], 0.0f};
}

// ========================
// 主渲染逻辑
// ========================
int main() {
  const int W = 512; // 图像宽度
  const int H = 512; // 图像高度

  // 加载模型和纹理
  // 注意：这里的模型和纹理路径需要根据实际情况调整

  Mesh mesh = loadModel("models/spot/spot_triangulated_good.obj");
  Texture diffuse_tex = loadTexture("models/spot/spot_texture.png");
  Texture bump_tex = loadTexture("models/spot/hmap.jpg");
  Texture disp_tex = loadTexture("models/bricks2/bricks2_disp.jpg");
  Texture parallax_tex = loadTexture("models/spot/hmap.jpg");

  Mesh rock_mesh = loadModel("models/rock/rock.obj");
  Texture rock_diffuse_tex = loadTexture("models/rock/rock.png");
  Texture rock_bump_tex = loadTexture("models/rock/rock.png");

  Mesh brick2_mesh = loadModel("models/bricks2/bricks2.obj");
  std::cout << "Loaded brick2 mesh with " << brick2_mesh.triangles.size()
            << " triangles." << std::endl;
  for (const auto &tri : brick2_mesh.triangles) {
    std::cout << "Triangle vertices: " << tri.v[0].pos << ", " << tri.v[1].pos
              << ", " << tri.v[2].pos << std::endl;

    std::cout << "Triangle normals: " << tri.v[0].normal << ", "
              << tri.v[1].normal << ", " << tri.v[2].normal << std::endl;

    std::cout << "Triangle UVs: " << tri.v[0].uv << ", " << tri.v[1].uv << ", "
              << tri.v[2].uv << std::endl;
  }
  Texture brick2_diffuse_tex = loadTexture("models/bricks2/bricks2.jpg");
  Texture brick2_normal_tex = loadTexture("models/bricks2/bricks2_normal.jpg");
  Texture brick2_disp_tex = loadTexture("models/bricks2/bricks2_disp.jpg");

  // 相机参数
  Vec3 eye_position = {-2, 2, -2};
  Vec3 target_position = {0, 0, 0};
  Vec3 up_direction = {0, 1, 0};

  // 构建视图矩阵和投影矩阵
  Mat4 view_matrix = Mat4::lookAt(eye_position, target_position, up_direction);
  float near_plane = 0.1f;
  float far_plane = 100.0f;
  Mat4 proj_matrix =
      Mat4::perspective(45.0f, W / (float)H, near_plane, far_plane);

  Mat4 NormalMatrix =
      view_matrix.inverse().transpose(); // 法线矩阵，用于法线变换

  // 定义光源用于MultiLightShader
  std::vector<Light> lights;
  lights.push_back({{2.0f, 2.0f, 2.0f},
                    {0.0f, 0.0f, 0.0f},
                    {1.0f, 0.0f, 0.0f},
                    1.0f}); // 红色点光源
  lights.push_back({{-2.0f, 3.0f, -1.0f},
                    {0.0f, 0.0f, 0.0f},
                    {0.0f, 0.0f, 1.0f},
                    0.8f}); // 蓝色点光源
  lights.push_back({{-2, -2, -2},
                    {0.0f, 0.0f, 0.0f},
                    {0.0f, 1.0f, 0.0f},
                    0.8f}); // 绿色点光源

  // 定义用于LightDepthShader的光源视图投影矩阵
  // 假设是一个平行光，从某个方向照射
  Vec3 light_pos = {3.0f, 3.0f, 3.0f};    // 光源位置
  Vec3 light_target = {0.0f, 0.0f, 0.0f}; // 光源看向原点
  Vec3 light_up = {0.0f, 1.0f, 0.0f};     // 光源向上方向
  Mat4 light_view_matrix = Mat4::lookAt(light_pos, light_target, light_up);
  Mat4 light_proj_matrix =
      Mat4::orthographic(-2.0f, 2.0f, -2.0f, 2.0f, 1.0f, 20.0f); // 正交投影
  Mat4 light_mvp = light_proj_matrix * light_view_matrix;

  // 定义要渲染的shader模式及其对应的输出文件名
  std::vector<std::pair<ShaderType, std::string>> shader_modes = {
      {ShaderType::BlinnPhong, "output_blinn.ppm"},
      {ShaderType::Normal, "output_normal.ppm"},
      {ShaderType::Texture, "output_texture.ppm"},
      {ShaderType::Bump, "output_bump.ppm"},
      {ShaderType::Displacement, "output_disp.ppm"},
      {ShaderType::Shadow, "output_shadow.ppm"},
      {ShaderType::AO, "output_ao.ppm"},
      {ShaderType::AlphaBlend, "output_alpha.ppm"},
      {ShaderType::Parallax, "output_parallax.ppm"}, // 新增视差贴图shader输出
      {ShaderType::SSAO, "output_ssao.ppm"},
      {ShaderType::SpotLight, "output_spotlight.ppm"},
      {ShaderType::Depth, "output_depth.ppm"},
      {ShaderType::LightDepth, "output_light_depth.ppm"},
      {ShaderType::MultiLight, "output_multi_light.ppm"},
      {ShaderType::NoMSAA, "output_nomsa.ppm"}, // 新增NoMSAA输出
      {ShaderType::Bloom, "output_bloom.ppm"}, // 新增 Bloom 着色器
  };

  // 遍历每种shader模式，进行渲染并输出图像
  for (auto &mode : shader_modes) {
    std::unique_ptr<IShader> shader; // 使用智能指针管理shader生命周期
    Mesh *current_mesh = &mesh;
    // 根据当前的shader类型动态创建对应的shader实例
    switch (mode.first) {
    case ShaderType::BlinnPhong:
      shader = std::make_unique<BlinnPhongShader>(
          proj_matrix * view_matrix, NormalMatrix, eye_position,
          Vec3{1.0f, 1.0f, -1.0f}, diffuse_tex);
      break;
    case ShaderType::Normal:
      shader = std::make_unique<NormalShader>(proj_matrix * view_matrix,
                                              NormalMatrix);
      break;
    case ShaderType::Texture:
      shader = std::make_unique<TextureShader>(proj_matrix * view_matrix,
                                               NormalMatrix, diffuse_tex);
      break;
    case ShaderType::Bump:
      shader = std::make_unique<BumpShader>(
          proj_matrix * view_matrix, NormalMatrix, eye_position,
          Vec3{1.0f, 1.0f, -1.0f}, rock_diffuse_tex, bump_tex);
      current_mesh = &rock_mesh;
      break;
    case ShaderType::Displacement:
      shader = std::make_unique<DisplacementShader>(
          proj_matrix * view_matrix, NormalMatrix, eye_position,
          Vec3{1.0f, 1.0f, -1.0f}, diffuse_tex, disp_tex, 0.1f);
      break;
    case ShaderType::Parallax:
      // 视差贴图shader，使用hmap.jpg作为视差贴图
      shader = std::make_unique<ParallaxShader>(
          proj_matrix * view_matrix, NormalMatrix, eye_position,
          Vec3{1.0f, 1.0f, -1.0f}, brick2_diffuse_tex, brick2_disp_tex,
          brick2_normal_tex, 0.05f); // 0.05f为视差scale，可调
      current_mesh = &brick2_mesh;   // 使用rock_mesh作为视差贴图的模型
      break;
    case ShaderType::Shadow:
      shader = std::make_unique<ShadowShader>(
          proj_matrix * view_matrix, NormalMatrix, Vec3{1.0f, 1.0f, -1.0f});
      break;
    case ShaderType::AO:
      // AO shader使用一个假设的环境光方向
      shader = std::make_unique<AOShader>(
          proj_matrix * view_matrix, NormalMatrix,
          Vec3{0.0f, 1.0f, 0.0f}); // 例如，环境光来自上方
      break;
    case ShaderType::AlphaBlend:
      shader = std::make_unique<AlphaBlendShader>(proj_matrix * view_matrix,
                                                  NormalMatrix, diffuse_tex,
                                                  0.7f); // 0.7f 透明度
      break;
    case ShaderType::SSAO:
      // SSAO shader需要视图矩阵和近远裁剪面距离，以及屏幕UV (用于噪声纹理)
      shader = std::make_unique<SSAOShader>(
          proj_matrix * view_matrix, NormalMatrix, view_matrix, near_plane,
          far_plane, 16); // 16个采样点
      break;
    case ShaderType::SpotLight:
      // 聚光灯光源位置和方向，内锥角和外锥角
      shader = std::make_unique<SpotLightShader>(
          proj_matrix * view_matrix, eye_position, Vec3{0.0f, 5.0f, 5.0f},
          Vec3{0.0f, -1.0f, -1.0f}, diffuse_tex, 5.0f, 10.0f);
      break;
    case ShaderType::Depth:
      shader = std::make_unique<DepthShader>(proj_matrix * view_matrix);
      break;
    case ShaderType::LightDepth:
      // 这个shader用于演示从光源视角看到的深度图
      shader = std::make_unique<LightDepthShader>(light_mvp);
      break;
    case ShaderType::MultiLight:
      shader = std::make_unique<MultiLightBlinnPhongShader>(
          proj_matrix * view_matrix, eye_position, lights, diffuse_tex);
      break;
    case ShaderType::Bloom:
      shader = std::make_unique<BloomShader>(proj_matrix * view_matrix, diffuse_tex, 0.8f);
      break;
    default:
      // 默认使用Blinn-Phong
      shader = std::make_unique<BlinnPhongShader>(
          proj_matrix * view_matrix, NormalMatrix, eye_position,
          Vec3{1.0f, 1.0f, -1.0f}, diffuse_tex);
      break;
    }

    // 设置MSAA参数：NoMSAA模式下强制1倍采样，其它模式用默认
    if (mode.first == ShaderType::NoMSAA) {
      MSAA_SAMPLES = MSAA_SAMPLES_1;
      MSAA_OFFSETS = MSAA_OFFSETS_1;
    } else {
      MSAA_SAMPLES = MSAA_SAMPLES_4;
      MSAA_OFFSETS = MSAA_OFFSETS_4;
    }

    // 为每个样本点分配颜色缓冲区和Z缓冲区
    std::vector<Vec3> color_buffer(W * H * MSAA_SAMPLES);
    std::vector<float> z_buffer(
        W * H * MSAA_SAMPLES,
        std::numeric_limits<float>::max()); // 初始化为最大浮点数

    // 遍历模型中的每个三角形
    for (const auto &tri : (*current_mesh).triangles) {
      // 顶点着色器阶段，适配ShaderIO
      Vec4 clip_coords[3];
      float inv_w[3];

      // 定义ShaderIO数组（最大结构体尺寸）
      BlinnPhongShaderIO io_blinn[3];
      NormalShaderIO io_normal[3];
      TextureShaderIO io_tex[3];
      BumpShaderIO io_bump[3];
      DisplacementShaderIO io_disp[3];
      ShadowShaderIO io_shadow[3];
      AOShaderIO io_ao[3];
      AlphaBlendShaderIO io_alpha[3];
      SSAOShaderIO io_ssao[3];
      SpotLightShaderIO io_spot[3];
      DepthShaderIO io_depth[3];
      LightDepthShaderIO io_lightdepth[3];
      MultiLightShaderIO io_multilight[3];
      BloomShaderIO io_bloom[3]; // 新增 BloomShaderIO

      IShaderIO *io_arr[3] = {nullptr, nullptr, nullptr};
      ParallaxShaderIO io_parallax[3];

      // 选择对应的IO类型
      switch (mode.first) {
      case ShaderType::BlinnPhong:
        for (int i = 0; i < 3; ++i) {
          io_blinn[i].position_world = tri.v[i].pos;
          io_blinn[i].normal_world = tri.v[i].normal;
          io_blinn[i].tangent_world = tri.v[i].tangent;
          io_blinn[i].uv = tri.v[i].uv;
          io_arr[i] = &io_blinn[i];
        }
        break;
      case ShaderType::Normal:
        for (int i = 0; i < 3; ++i) {
          io_normal[i].position_world = tri.v[i].pos;
          io_normal[i].normal_world = tri.v[i].normal;
          io_arr[i] = &io_normal[i];
        }
        break;
      case ShaderType::Texture:
        for (int i = 0; i < 3; ++i) {
          io_tex[i].position_world = tri.v[i].pos;
          io_tex[i].uv = tri.v[i].uv;
          io_arr[i] = &io_tex[i];
        }
        break;
      case ShaderType::Bump:
        for (int i = 0; i < 3; ++i) {
          io_bump[i].position_world = tri.v[i].pos;
          io_bump[i].normal_world = tri.v[i].normal;
          io_bump[i].tangent_world = tri.v[i].tangent;
          io_bump[i].uv = tri.v[i].uv;
          io_arr[i] = &io_bump[i];
        }
        break;
      case ShaderType::Displacement:
        for (int i = 0; i < 3; ++i) {
          io_disp[i].position_world = tri.v[i].pos;
          io_disp[i].normal_world = tri.v[i].normal;
          io_disp[i].tangent_world = tri.v[i].tangent;
          io_disp[i].uv = tri.v[i].uv;
          io_arr[i] = &io_disp[i];
        }
        break;
      case ShaderType::Parallax:
        for (int i = 0; i < 3; ++i) {
          io_parallax[i].position_world = tri.v[i].pos;
          io_parallax[i].normal_world = tri.v[i].normal;
          io_parallax[i].tangent_world = tri.v[i].tangent;
          io_parallax[i].uv = tri.v[i].uv;
          io_arr[i] = &io_parallax[i];
        }
        break;
      case ShaderType::Shadow:
        for (int i = 0; i < 3; ++i) {
          io_shadow[i].position_world = tri.v[i].pos;
          io_shadow[i].normal_world = tri.v[i].normal;
          io_arr[i] = &io_shadow[i];
        }
        break;
      case ShaderType::AO:
        for (int i = 0; i < 3; ++i) {
          io_ao[i].position_world = tri.v[i].pos;
          io_ao[i].normal_world = tri.v[i].normal;
          io_arr[i] = &io_ao[i];
        }
        break;
      case ShaderType::AlphaBlend:
        for (int i = 0; i < 3; ++i) {
          io_alpha[i].position_world = tri.v[i].pos;
          io_alpha[i].uv = tri.v[i].uv;
          io_arr[i] = &io_alpha[i];
        }
        break;
      case ShaderType::SSAO:
        for (int i = 0; i < 3; ++i) {
          io_ssao[i].position_world = tri.v[i].pos;
          io_ssao[i].normal_world = tri.v[i].normal;
          io_arr[i] = &io_ssao[i];
        }
        break;
      case ShaderType::SpotLight:
        for (int i = 0; i < 3; ++i) {
          io_spot[i].position_world = tri.v[i].pos;
          io_spot[i].normal_world = tri.v[i].normal;
          io_spot[i].uv = tri.v[i].uv;
          io_arr[i] = &io_spot[i];
        }
        break;
      case ShaderType::Depth:
        for (int i = 0; i < 3; ++i) {
          io_depth[i].position_world = tri.v[i].pos;
          io_arr[i] = &io_depth[i];
        }
        break;
      case ShaderType::LightDepth:
        for (int i = 0; i < 3; ++i) {
          io_lightdepth[i].position_world = tri.v[i].pos;
          io_arr[i] = &io_lightdepth[i];
        }
        break;
      case ShaderType::MultiLight:
        for (int i = 0; i < 3; ++i) {
          io_multilight[i].position_world = tri.v[i].pos;
          io_multilight[i].normal_world = tri.v[i].normal;
          io_multilight[i].uv = tri.v[i].uv;
          io_arr[i] = &io_multilight[i];
        }
        break;
      case ShaderType::Bloom:
        for (int i = 0; i < 3; ++i) {
          io_bloom[i].position_world = tri.v[i].pos;
          io_bloom[i].uv = tri.v[i].uv;
          io_arr[i] = &io_bloom[i];
        }
        break;
      default:
        for (int i = 0; i < 3; ++i) {
          io_blinn[i].position_world = tri.v[i].pos;
          io_blinn[i].normal_world = tri.v[i].normal;
          io_blinn[i].tangent_world = tri.v[i].tangent;
          io_blinn[i].uv = tri.v[i].uv;
          io_arr[i] = &io_blinn[i];
        }
        break;
      }

      // 顶点着色器调用
      for (int i = 0; i < 3; ++i) {
        shader->vertex(*io_arr[i]);
        // 取裁剪空间坐标
        switch (mode.first) {
        case ShaderType::BlinnPhong:
          clip_coords[i] = io_blinn[i].position_clip;
          break;
        case ShaderType::Normal:
          clip_coords[i] = io_normal[i].position_clip;
          break;
        case ShaderType::Texture:
          clip_coords[i] = io_tex[i].position_clip;
          break;
        case ShaderType::Bump:
          clip_coords[i] = io_bump[i].position_clip;
          break;
        case ShaderType::Displacement:
          clip_coords[i] = io_disp[i].position_clip;
          break;
        case ShaderType::Parallax:
          clip_coords[i] = io_parallax[i].position_clip;
          break;
        case ShaderType::Shadow:
          clip_coords[i] = io_shadow[i].position_clip;
          break;
        case ShaderType::AO:
          clip_coords[i] = io_ao[i].position_clip;
          break;
        case ShaderType::AlphaBlend:
          clip_coords[i] = io_alpha[i].position_clip;
          break;
        case ShaderType::SSAO:
          clip_coords[i] = io_ssao[i].position_clip;
          break;
        case ShaderType::SpotLight:
          clip_coords[i] = io_spot[i].position_clip;
          break;
        case ShaderType::Depth:
          clip_coords[i] = io_depth[i].position_clip;
          break;
        case ShaderType::LightDepth:
          clip_coords[i] = io_lightdepth[i].position_clip;
          break;
        case ShaderType::MultiLight:
          clip_coords[i] = io_multilight[i].position_clip;
          break;
        case ShaderType::Bloom:
          clip_coords[i] = io_bloom[i].position_clip;
          break;
        default:
          clip_coords[i] = io_blinn[i].position_clip;
          break;
        }
        inv_w[i] = 1.0f / clip_coords[i].w;
      }

      // 三角形光栅化阶段
      // 使用Lambda函数封装光栅化逻辑，方便内部调用
      auto rasterize_triangle = [&](const Triangle &original_tri,
                                    IShader &current_shader, int img_W,
                                    int img_H, int samples_per_pixel,
                                    const float msaa_offsets[][2],
                                    const Vec4 clip_coords_arr[3],
                                    const float inv_w_arr[3],
                                    std::vector<Vec3> &current_color_buffer,
                                    std::vector<float> &current_z_buffer,
                                    IShaderIO *io_arr[3]) {
        Vec3 screen_coords[3];
        for (int i = 0; i < 3; ++i) {
          screen_coords[i] = ndcToScreen(
              clip_coords_arr[i] / clip_coords_arr[i].w, img_W, img_H);
        }
        float area =
            edgeFunc(screen_coords[0], screen_coords[1], screen_coords[2]);
        if (area < 1.0f && area > -1.0f)
          return;
        if (area < 0.0f)
          return;
        float min_x = std::min(
            {screen_coords[0].x, screen_coords[1].x, screen_coords[2].x});
        float max_x = std::max(
            {screen_coords[0].x, screen_coords[1].x, screen_coords[2].x});
        float min_y = std::min(
            {screen_coords[0].y, screen_coords[1].y, screen_coords[2].y});
        float max_y = std::max(
            {screen_coords[0].y, screen_coords[1].y, screen_coords[2].y});
        int x_start = std::max(0, (int)std::floor(min_x));
        int x_end = std::min(img_W - 1, (int)std::ceil(max_x));
        int y_start = std::max(0, (int)std::floor(min_y));
        int y_end = std::min(img_H - 1, (int)std::ceil(max_y));
        for (int y = y_start; y <= y_end; ++y) {
          for (int x = x_start; x <= x_end; ++x) {
            for (int s = 0; s < samples_per_pixel; ++s) {
              Vec3 p = getSamplePoint(x, y, s, msaa_offsets);
              float w0 = edgeFunc(screen_coords[1], screen_coords[2], p);
              float w1 = edgeFunc(screen_coords[2], screen_coords[0], p);
              float w2 = edgeFunc(screen_coords[0], screen_coords[1], p);
              if (w0 >= -std::numeric_limits<float>::epsilon() &&
                  w1 >= -std::numeric_limits<float>::epsilon() &&
                  w2 >= -std::numeric_limits<float>::epsilon()) {
                float inv_total_area = 1.0f / area;
                w0 *= inv_total_area;
                w1 *= inv_total_area;
                w2 *= inv_total_area;
                float interpolated_inv_w = baryInterp(
                    inv_w_arr[0], inv_w_arr[1], inv_w_arr[2], w0, w1, w2);
                float interpolated_w = 1.0f / interpolated_inv_w;
                float z_ndc_0 = clip_coords_arr[0].z * inv_w_arr[0];
                float z_ndc_1 = clip_coords_arr[1].z * inv_w_arr[1];
                float z_ndc_2 = clip_coords_arr[2].z * inv_w_arr[2];
                float z_ndc = baryInterp(z_ndc_0, z_ndc_1, z_ndc_2, w0, w1, w2);
                int buffer_idx = (y * img_W + x) * samples_per_pixel + s;
                if (z_ndc < current_z_buffer[buffer_idx]) {
                  current_z_buffer[buffer_idx] = z_ndc;
                  // 插值 ShaderIO
                  IShaderIO *io_frag = nullptr;
                  switch (mode.first) {
                  case ShaderType::BlinnPhong: {
                    static BlinnPhongShaderIO frag;
                    frag.position_world =
                        baryInterp(
                            ((BlinnPhongShaderIO *)io_arr[0])->position_world *
                                inv_w_arr[0],
                            ((BlinnPhongShaderIO *)io_arr[1])->position_world *
                                inv_w_arr[1],
                            ((BlinnPhongShaderIO *)io_arr[2])->position_world *
                                inv_w_arr[2],
                            w0, w1, w2) *
                        interpolated_w;
                    frag.normal_world =
                        baryInterp(
                            ((BlinnPhongShaderIO *)io_arr[0])->normal_world *
                                inv_w_arr[0],
                            ((BlinnPhongShaderIO *)io_arr[1])->normal_world *
                                inv_w_arr[1],
                            ((BlinnPhongShaderIO *)io_arr[2])->normal_world *
                                inv_w_arr[2],
                            w0, w1, w2) *
                        interpolated_w;
                    frag.tangent_world =
                        baryInterp(
                            ((BlinnPhongShaderIO *)io_arr[0])->tangent_world *
                                inv_w_arr[0],
                            ((BlinnPhongShaderIO *)io_arr[1])->tangent_world *
                                inv_w_arr[1],
                            ((BlinnPhongShaderIO *)io_arr[2])->tangent_world *
                                inv_w_arr[2],
                            w0, w1, w2) *
                        interpolated_w;
                    frag.uv = baryInterp(((BlinnPhongShaderIO *)io_arr[0])->uv *
                                             inv_w_arr[0],
                                         ((BlinnPhongShaderIO *)io_arr[1])->uv *
                                             inv_w_arr[1],
                                         ((BlinnPhongShaderIO *)io_arr[2])->uv *
                                             inv_w_arr[2],
                                         w0, w1, w2) *
                              interpolated_w;
                    frag.ndc_z = z_ndc;
                    io_frag = &frag;
                    break;
                  }
                  case ShaderType::Normal: {
                    static NormalShaderIO frag;
                    frag.position_world =
                        baryInterp(
                            ((NormalShaderIO *)io_arr[0])->position_world *
                                inv_w_arr[0],
                            ((NormalShaderIO *)io_arr[1])->position_world *
                                inv_w_arr[1],
                            ((NormalShaderIO *)io_arr[2])->position_world *
                                inv_w_arr[2],
                            w0, w1, w2) *
                        interpolated_w;
                    frag.normal_world =
                        baryInterp(((NormalShaderIO *)io_arr[0])->normal_world *
                                       inv_w_arr[0],
                                   ((NormalShaderIO *)io_arr[1])->normal_world *
                                       inv_w_arr[1],
                                   ((NormalShaderIO *)io_arr[2])->normal_world *
                                       inv_w_arr[2],
                                   w0, w1, w2) *
                        interpolated_w;
                    io_frag = &frag;
                    break;
                  }
                  case ShaderType::Texture: {
                    static TextureShaderIO frag;
                    frag.position_world =
                        baryInterp(
                            ((TextureShaderIO *)io_arr[0])->position_world *
                                inv_w_arr[0],
                            ((TextureShaderIO *)io_arr[1])->position_world *
                                inv_w_arr[1],
                            ((TextureShaderIO *)io_arr[2])->position_world *
                                inv_w_arr[2],
                            w0, w1, w2) *
                        interpolated_w;
                    frag.uv =
                        baryInterp(
                            ((TextureShaderIO *)io_arr[0])->uv * inv_w_arr[0],
                            ((TextureShaderIO *)io_arr[1])->uv * inv_w_arr[1],
                            ((TextureShaderIO *)io_arr[2])->uv * inv_w_arr[2],
                            w0, w1, w2) *
                        interpolated_w;
                    io_frag = &frag;
                    break;
                  }
                  case ShaderType::Bump: {
                    static BumpShaderIO frag;
                    frag.position_world =
                        baryInterp(((BumpShaderIO *)io_arr[0])->position_world *
                                       inv_w_arr[0],
                                   ((BumpShaderIO *)io_arr[1])->position_world *
                                       inv_w_arr[1],
                                   ((BumpShaderIO *)io_arr[2])->position_world *
                                       inv_w_arr[2],
                                   w0, w1, w2) *
                        interpolated_w;
                    frag.normal_world =
                        baryInterp(((BumpShaderIO *)io_arr[0])->normal_world *
                                       inv_w_arr[0],
                                   ((BumpShaderIO *)io_arr[1])->normal_world *
                                       inv_w_arr[1],
                                   ((BumpShaderIO *)io_arr[2])->normal_world *
                                       inv_w_arr[2],
                                   w0, w1, w2) *
                        interpolated_w;
                    frag.tangent_world =
                        baryInterp(((BumpShaderIO *)io_arr[0])->tangent_world *
                                       inv_w_arr[0],
                                   ((BumpShaderIO *)io_arr[1])->tangent_world *
                                       inv_w_arr[1],
                                   ((BumpShaderIO *)io_arr[2])->tangent_world *
                                       inv_w_arr[2],
                                   w0, w1, w2) *
                        interpolated_w;
                    frag.uv =
                        baryInterp(
                            ((BumpShaderIO *)io_arr[0])->uv * inv_w_arr[0],
                            ((BumpShaderIO *)io_arr[1])->uv * inv_w_arr[1],
                            ((BumpShaderIO *)io_arr[2])->uv * inv_w_arr[2], w0,
                            w1, w2) *
                        interpolated_w;
                    io_frag = &frag;
                    break;
                  }
                  case ShaderType::Displacement: {
                    static DisplacementShaderIO frag;
                    frag.position_world =
                        baryInterp(((DisplacementShaderIO *)io_arr[0])
                                           ->position_world *
                                       inv_w_arr[0],
                                   ((DisplacementShaderIO *)io_arr[1])
                                           ->position_world *
                                       inv_w_arr[1],
                                   ((DisplacementShaderIO *)io_arr[2])
                                           ->position_world *
                                       inv_w_arr[2],
                                   w0, w1, w2) *
                        interpolated_w;
                    frag.normal_world =
                        baryInterp(
                            ((DisplacementShaderIO *)io_arr[0])->normal_world *
                                inv_w_arr[0],
                            ((DisplacementShaderIO *)io_arr[1])->normal_world *
                                inv_w_arr[1],
                            ((DisplacementShaderIO *)io_arr[2])->normal_world *
                                inv_w_arr[2],
                            w0, w1, w2) *
                        interpolated_w;
                    frag.tangent_world =
                        baryInterp(
                            ((DisplacementShaderIO *)io_arr[0])->tangent_world *
                                inv_w_arr[0],
                            ((DisplacementShaderIO *)io_arr[1])->tangent_world *
                                inv_w_arr[1],
                            ((DisplacementShaderIO *)io_arr[2])->tangent_world *
                                inv_w_arr[2],
                            w0, w1, w2) *
                        interpolated_w;
                    frag.uv =
                        baryInterp(((DisplacementShaderIO *)io_arr[0])->uv *
                                       inv_w_arr[0],
                                   ((DisplacementShaderIO *)io_arr[1])->uv *
                                       inv_w_arr[1],
                                   ((DisplacementShaderIO *)io_arr[2])->uv *
                                       inv_w_arr[2],
                                   w0, w1, w2) *
                        interpolated_w;
                    io_frag = &frag;
                    break;
                  }
                  case ShaderType::Parallax: {
                    static ParallaxShaderIO frag;
                    frag.position_world =
                        baryInterp(
                            ((ParallaxShaderIO *)io_arr[0])->position_world *
                                inv_w_arr[0],
                            ((ParallaxShaderIO *)io_arr[1])->position_world *
                                inv_w_arr[1],
                            ((ParallaxShaderIO *)io_arr[2])->position_world *
                                inv_w_arr[2],
                            w0, w1, w2) *
                        interpolated_w;
                    frag.normal_world =
                        baryInterp(
                            ((ParallaxShaderIO *)io_arr[0])->normal_world *
                                inv_w_arr[0],
                            ((ParallaxShaderIO *)io_arr[1])->normal_world *
                                inv_w_arr[1],
                            ((ParallaxShaderIO *)io_arr[2])->normal_world *
                                inv_w_arr[2],
                            w0, w1, w2) *
                        interpolated_w;
                    frag.tangent_world =
                        baryInterp(
                            ((ParallaxShaderIO *)io_arr[0])->tangent_world *
                                inv_w_arr[0],
                            ((ParallaxShaderIO *)io_arr[1])->tangent_world *
                                inv_w_arr[1],
                            ((ParallaxShaderIO *)io_arr[2])->tangent_world *
                                inv_w_arr[2],
                            w0, w1, w2) *
                        interpolated_w;
                    frag.uv =
                        baryInterp(
                            ((ParallaxShaderIO *)io_arr[0])->uv * inv_w_arr[0],
                            ((ParallaxShaderIO *)io_arr[1])->uv * inv_w_arr[1],
                            ((ParallaxShaderIO *)io_arr[2])->uv * inv_w_arr[2],
                            w0, w1, w2) *
                        interpolated_w;
                    io_frag = &frag;
                    break;
                  }
                  case ShaderType::Shadow: {
                    static ShadowShaderIO frag;
                    frag.position_world =
                        baryInterp(
                            ((ShadowShaderIO *)io_arr[0])->position_world *
                                inv_w_arr[0],
                            ((ShadowShaderIO *)io_arr[1])->position_world *
                                inv_w_arr[1],
                            ((ShadowShaderIO *)io_arr[2])->position_world *
                                inv_w_arr[2],
                            w0, w1, w2) *
                        interpolated_w;
                    frag.normal_world =
                        baryInterp(((ShadowShaderIO *)io_arr[0])->normal_world *
                                       inv_w_arr[0],
                                   ((ShadowShaderIO *)io_arr[1])->normal_world *
                                       inv_w_arr[1],
                                   ((ShadowShaderIO *)io_arr[2])->normal_world *
                                       inv_w_arr[2],
                                   w0, w1, w2) *
                        interpolated_w;
                    io_frag = &frag;
                    break;
                  }
                  case ShaderType::AO: {
                    static AOShaderIO frag;
                    frag.position_world =
                        baryInterp(((AOShaderIO *)io_arr[0])->position_world *
                                       inv_w_arr[0],
                                   ((AOShaderIO *)io_arr[1])->position_world *
                                       inv_w_arr[1],
                                   ((AOShaderIO *)io_arr[2])->position_world *
                                       inv_w_arr[2],
                                   w0, w1, w2) *
                        interpolated_w;
                    frag.normal_world =
                        baryInterp(((AOShaderIO *)io_arr[0])->normal_world *
                                       inv_w_arr[0],
                                   ((AOShaderIO *)io_arr[1])->normal_world *
                                       inv_w_arr[1],
                                   ((AOShaderIO *)io_arr[2])->normal_world *
                                       inv_w_arr[2],
                                   w0, w1, w2) *
                        interpolated_w;
                    io_frag = &frag;
                    break;
                  }
                  case ShaderType::AlphaBlend: {
                    static AlphaBlendShaderIO frag;
                    frag.position_world =
                        baryInterp(
                            ((AlphaBlendShaderIO *)io_arr[0])->position_world *
                                inv_w_arr[0],
                            ((AlphaBlendShaderIO *)io_arr[1])->position_world *
                                inv_w_arr[1],
                            ((AlphaBlendShaderIO *)io_arr[2])->position_world *
                                inv_w_arr[2],
                            w0, w1, w2) *
                        interpolated_w;
                    frag.uv = baryInterp(((AlphaBlendShaderIO *)io_arr[0])->uv *
                                             inv_w_arr[0],
                                         ((AlphaBlendShaderIO *)io_arr[1])->uv *
                                             inv_w_arr[1],
                                         ((AlphaBlendShaderIO *)io_arr[2])->uv *
                                             inv_w_arr[2],
                                         w0, w1, w2) *
                              interpolated_w;
                    io_frag = &frag;
                    break;
                  }
                  case ShaderType::SSAO: {
                    static SSAOShaderIO frag;
                    frag.position_world =
                        baryInterp(((SSAOShaderIO *)io_arr[0])->position_world *
                                       inv_w_arr[0],
                                   ((SSAOShaderIO *)io_arr[1])->position_world *
                                       inv_w_arr[1],
                                   ((SSAOShaderIO *)io_arr[2])->position_world *
                                       inv_w_arr[2],
                                   w0, w1, w2) *
                        interpolated_w;
                    frag.normal_world =
                        baryInterp(((SSAOShaderIO *)io_arr[0])->normal_world *
                                       inv_w_arr[0],
                                   ((SSAOShaderIO *)io_arr[1])->normal_world *
                                       inv_w_arr[1],
                                   ((SSAOShaderIO *)io_arr[2])->normal_world *
                                       inv_w_arr[2],
                                   w0, w1, w2) *
                        interpolated_w;
                    frag.uv_screen_normalized =
                        baryInterp(Vec2{screen_coords[0].x / img_W,
                                        screen_coords[0].y / img_H} *
                                       inv_w_arr[0],
                                   Vec2{screen_coords[1].x / img_W,
                                        screen_coords[1].y / img_H} *
                                       inv_w_arr[1],
                                   Vec2{screen_coords[2].x / img_W,
                                        screen_coords[2].y / img_H} *
                                       inv_w_arr[2],
                                   w0, w1, w2) *
                        interpolated_w;
                    io_frag = &frag;
                    break;
                  }
                  case ShaderType::SpotLight: {
                    static SpotLightShaderIO frag;
                    frag.position_world =
                        baryInterp(
                            ((SpotLightShaderIO *)io_arr[0])->position_world *
                                inv_w_arr[0],
                            ((SpotLightShaderIO *)io_arr[1])->position_world *
                                inv_w_arr[1],
                            ((SpotLightShaderIO *)io_arr[2])->position_world *
                                inv_w_arr[2],
                            w0, w1, w2) *
                        interpolated_w;
                    frag.normal_world =
                        baryInterp(
                            ((SpotLightShaderIO *)io_arr[0])->normal_world *
                                inv_w_arr[0],
                            ((SpotLightShaderIO *)io_arr[1])->normal_world *
                                inv_w_arr[1],
                            ((SpotLightShaderIO *)io_arr[2])->normal_world *
                                inv_w_arr[2],
                            w0, w1, w2) *
                        interpolated_w;
                    frag.uv =
                        baryInterp(
                            ((SpotLightShaderIO *)io_arr[0])->uv * inv_w_arr[0],
                            ((SpotLightShaderIO *)io_arr[1])->uv * inv_w_arr[1],
                            ((SpotLightShaderIO *)io_arr[2])->uv * inv_w_arr[2],
                            w0, w1, w2) *
                        interpolated_w;
                    io_frag = &frag;
                    break;
                  }
                  case ShaderType::Depth: {
                    static DepthShaderIO frag;
                    frag.position_world =
                        baryInterp(
                            ((DepthShaderIO *)io_arr[0])->position_world *
                                inv_w_arr[0],
                            ((DepthShaderIO *)io_arr[1])->position_world *
                                inv_w_arr[1],
                            ((DepthShaderIO *)io_arr[2])->position_world *
                                inv_w_arr[2],
                            w0, w1, w2) *
                        interpolated_w;
                    frag.ndc_z = z_ndc;
                    io_frag = &frag;
                    break;
                  }
                  case ShaderType::LightDepth: {
                    static LightDepthShaderIO frag;
                    frag.position_world =
                        baryInterp(
                            ((LightDepthShaderIO *)io_arr[0])->position_world *
                                inv_w_arr[0],
                            ((LightDepthShaderIO *)io_arr[1])->position_world *
                                inv_w_arr[1],
                            ((LightDepthShaderIO *)io_arr[2])->position_world *
                                inv_w_arr[2],
                            w0, w1, w2) *
                        interpolated_w;
                    frag.ndc_z = z_ndc;
                    io_frag = &frag;
                    break;
                  }
                  case ShaderType::MultiLight: {
                    static MultiLightShaderIO frag;
                    frag.position_world =
                        baryInterp(
                            ((MultiLightShaderIO *)io_arr[0])->position_world *
                                inv_w_arr[0],
                            ((MultiLightShaderIO *)io_arr[1])->position_world *
                                inv_w_arr[1],
                            ((MultiLightShaderIO *)io_arr[2])->position_world *
                                inv_w_arr[2],
                            w0, w1, w2) *
                        interpolated_w;
                    frag.normal_world =
                        baryInterp(
                            ((MultiLightShaderIO *)io_arr[0])->normal_world *
                                inv_w_arr[0],
                            ((MultiLightShaderIO *)io_arr[1])->normal_world *
                                inv_w_arr[1],
                            ((MultiLightShaderIO *)io_arr[2])->normal_world *
                                inv_w_arr[2],
                            w0, w1, w2) *
                        interpolated_w;
                    frag.uv = baryInterp(((MultiLightShaderIO *)io_arr[0])->uv *
                                             inv_w_arr[0],
                                         ((MultiLightShaderIO *)io_arr[1])->uv *
                                             inv_w_arr[1],
                                         ((MultiLightShaderIO *)io_arr[2])->uv *
                                             inv_w_arr[2],
                                         w0, w1, w2) *
                              interpolated_w;
                    io_frag = &frag;
                    break;
                  }
                  case ShaderType::Bloom: {
                    static BloomShaderIO frag;
                    frag.position_world = baryInterp(((BloomShaderIO *)io_arr[0])->position_world * inv_w_arr[0],
                                                    ((BloomShaderIO *)io_arr[1])->position_world * inv_w_arr[1],
                                                    ((BloomShaderIO *)io_arr[2])->position_world * inv_w_arr[2],
                                                    w0, w1, w2) * interpolated_w;
                    frag.uv = baryInterp(((BloomShaderIO *)io_arr[0])->uv * inv_w_arr[0],
                                        ((BloomShaderIO *)io_arr[1])->uv * inv_w_arr[1],
                                        ((BloomShaderIO *)io_arr[2])->uv * inv_w_arr[2],
                                        w0, w1, w2) * interpolated_w;
                    io_frag = &frag;
                    break;
                  }
                    default: {
                      static BlinnPhongShaderIO frag;
                      frag.position_world =
                          baryInterp(
                              ((BlinnPhongShaderIO*)io_arr[0])->position_world *
                                  inv_w_arr[0],
                              ((BlinnPhongShaderIO*)io_arr[1])->position_world *
                                  inv_w_arr[1],
                              ((BlinnPhongShaderIO*)io_arr[2])->position_world *
                                  inv_w_arr[2],
                              w0, w1, w2) *
                          interpolated_w;
                      frag.normal_world =
                          baryInterp(
                              ((BlinnPhongShaderIO*)io_arr[0])->normal_world *
                                  inv_w_arr[0],
                              ((BlinnPhongShaderIO*)io_arr[1])->normal_world *
                                  inv_w_arr[1],
                              ((BlinnPhongShaderIO*)io_arr[2])->normal_world *
                                  inv_w_arr[2],
                              w0, w1, w2) *
                          interpolated_w;
                      frag.tangent_world =
                          baryInterp(
                              ((BlinnPhongShaderIO*)io_arr[0])->tangent_world *
                                  inv_w_arr[0],
                              ((BlinnPhongShaderIO*)io_arr[1])->tangent_world *
                                  inv_w_arr[1],
                              ((BlinnPhongShaderIO*)io_arr[2])->tangent_world *
                                  inv_w_arr[2],
                              w0, w1, w2) *
                          interpolated_w;
                      frag.uv =
                          baryInterp(((BlinnPhongShaderIO*)io_arr[0])->uv *
                                         inv_w_arr[0],
                                     ((BlinnPhongShaderIO*)io_arr[1])->uv *
                                         inv_w_arr[1],
                                     ((BlinnPhongShaderIO*)io_arr[2])->uv *
                                         inv_w_arr[2],
                                     w0, w1, w2) *
                          interpolated_w;
                      frag.ndc_z = z_ndc;
                      io_frag = &frag;
                      break;
                    }
                  }
                  current_shader.fragment(*io_frag);
                  // 取颜色
                  Vec3 out_color;
                  switch (mode.first) {
                  case ShaderType::BlinnPhong:
                    out_color = ((BlinnPhongShaderIO *)io_frag)->color;
                    break;
                  case ShaderType::Normal:
                    out_color = ((NormalShaderIO *)io_frag)->color;
                    break;
                  case ShaderType::Texture:
                    out_color = ((TextureShaderIO *)io_frag)->color;
                    break;
                  case ShaderType::Bump:
                    out_color = ((BumpShaderIO *)io_frag)->color;
                    break;
                  case ShaderType::Displacement:
                    out_color = ((DisplacementShaderIO *)io_frag)->color;
                    break;
                  case ShaderType::Parallax:
                    out_color = ((ParallaxShaderIO *)io_frag)->color;
                    break;
                  case ShaderType::Shadow:
                    out_color = ((ShadowShaderIO *)io_frag)->color;
                    break;
                  case ShaderType::AO:
                    out_color = ((AOShaderIO *)io_frag)->color;
                    break;
                  case ShaderType::AlphaBlend:
                    out_color = ((AlphaBlendShaderIO *)io_frag)->color;
                    break;
                  case ShaderType::SSAO:
                    out_color = ((SSAOShaderIO *)io_frag)->color;
                    break;
                  case ShaderType::SpotLight:
                    out_color = ((SpotLightShaderIO *)io_frag)->color;
                    break;
                  case ShaderType::Depth:
                    out_color = ((DepthShaderIO *)io_frag)->color;
                    break;
                  case ShaderType::LightDepth:
                    out_color = ((LightDepthShaderIO *)io_frag)->color;
                    break;
                  case ShaderType::MultiLight:
                    out_color = ((MultiLightShaderIO *)io_frag)->color;
                    break;
                  case ShaderType::Bloom:
                    out_color = ((BloomShaderIO *)io_frag)->bloom_color;
                    break;
                  default:
                    out_color = ((BlinnPhongShaderIO *)io_frag)->color;
                    break;
                  }
                  current_color_buffer[buffer_idx] = out_color;
                }
              }
            }
          }
        }
      };
      rasterize_triangle(tri, *shader, W, H, MSAA_SAMPLES, MSAA_OFFSETS,
                         clip_coords, inv_w, color_buffer, z_buffer, io_arr);
    }

    std::string output_folder = "output_images/"; // 定义输出文件夹名称
    
    // 确保输出文件夹存在，如果不存在则创建它
    // std::filesystem::create_directories 会创建所有不存在的父目录
    try {
        if (!std::filesystem::exists(output_folder)) {
            std::filesystem::create_directories(output_folder);
            std::cout << "Created output directory: " << output_folder << std::endl;
        }
    } catch (const std::filesystem::filesystem_error& e) {
        std::cerr << "Error creating directory: " << e.what() << std::endl;
        return 1; // 错误退出
    }

    // 输出PPM图像
    std::string output_filepath = output_folder + mode.second; 
    std::ofstream ofs(output_filepath);
    if (!ofs.is_open()) {
      std::cerr << "Error: Could not open file " << mode.second
                << " for writing." << std::endl;
      continue;
    }
    ofs << "P3\n" << W << " " << H << "\n255\n"; // PPM文件头

    // 遍历每个像素，进行MSAA颜色混合
    for (int y = 0; y < H; ++y) {
      for (int x = 0; x < W; ++x) {
        Vec3 final_color_avg{};
        // 将所有采样点的颜色累加
        for (int s = 0; s < MSAA_SAMPLES; ++s) {
          final_color_avg =
              final_color_avg + color_buffer[(y * W + x) * MSAA_SAMPLES + s];
        }
        // 平均颜色
        final_color_avg = final_color_avg * (1.0f / MSAA_SAMPLES);

        // 钳制颜色值到[0, 1]范围，并转换为0-255整数
        int ir = static_cast<int>(
            std::max(0.0f, std::min(1.0f, final_color_avg.x)) * 255.0f + 0.5f);
        int ig = static_cast<int>(
            std::max(0.0f, std::min(1.0f, final_color_avg.y)) * 255.0f + 0.5f);
        int ib = static_cast<int>(
            std::max(0.0f, std::min(1.0f, final_color_avg.z)) * 255.0f + 0.5f);
        ofs << ir << " " << ig << " " << ib << "\n";
      }
    }
    ofs.close();
    std::cout << "Done: " << mode.second << std::endl;
  }

  return 0;
}