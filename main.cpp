#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional> // For std::function
#include <iostream>
#include <limits>
#include <memory>
#include <numbers>
#include <random>
#include <string>
// #include <type_traits> // For std::is_same_v
#include <vector>

#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

// ========================
// 基础数学结构: Vec3, Vec2, Vec4, Mat4
// ========================
struct Vec3 {
  float x, y, z;
  Vec3() : x(0.0f), y(0.0f), z(0.0f) {}
  Vec3(float _x, float _y, float _z) : x(_x), y(_y), z(_z) {}
  Vec3 operator+(const Vec3 &o) const { return {x + o.x, y + o.y, z + o.z}; }
  Vec3 operator-(const Vec3 &o) const { return {x - o.x, y - o.y, z - o.z}; }
  Vec3 operator*(float s) const { return {x * s, y * s, z * s}; }
  Vec3 operator/(float s) const {
    if (s == 0)
      return {0, 0, 0};
    return {x / s, y / s, z / s};
  }
  Vec3 operator-() const { return {-x, -y, -z}; }
  Vec3 operator*(const Vec3 &o) const { return {x * o.x, y * o.y, z * o.z}; }
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
  float dot(const Vec3 &o) const { return x * o.x + y * o.y + z * o.z; }
  Vec3 cross(const Vec3 &o) const {
    return {y * o.z - z * o.y, z * o.x - x * o.z, x * o.y - y * o.x};
  }
  float length() const { return std::sqrt(x * x + y * y + z * z); }
  Vec3 normalize() const {
    float l = length();
    if (l < 1e-6)
      return {0, 0, 0};
    return {x / l, y / l, z / l};
  }
};
struct Vec2 {
  float u, v;
  Vec2() : u(0.0f), v(0.0f) {}
  Vec2(float _u, float _v) : u(_u), v(_v) {}
  Vec2 operator=(const Vec2 &o) {
    u = o.u;
    v = o.v;
    return *this;
  }
  Vec2 operator*(float s) const { return {u * s, v * s}; }
  Vec2 operator+(const Vec2 &o) const { return {u + o.u, v + o.v}; }
  Vec2 operator-(const Vec2 &o) const { return {u - o.u, v - o.v}; }
  Vec2 operator-=(const Vec2 &o) {
    u -= o.u;
    v -= o.v;
    return *this;
  }
  Vec2 operator/(float s) const {
    if (s == 0)
      return {0, 0};
    return {u / s, v / s};
  }
  friend std::ostream &operator<<(std::ostream &os, const Vec2 &v) {
    os << "Vec2(" << v.u << ", " << v.v << ")";
    return os;
  }
};
struct Vec4 {
  float x, y, z, w;
  Vec4() : x(0.0f), y(0.0f), z(0.0f), w(0.0f) {}
  Vec4(float _x, float _y, float _z, float _w) : x(_x), y(_y), z(_z), w(_w) {}
  Vec4 operator/(float s) const {
    if (s == 0)
      return {0, 0, 0, 0};
    return {x / s, y / s, z / s, w / s};
  }
  Vec4 operator*(float s) const { return {x * s, y * s, z * s, w * s}; }
  Vec4 normalize() const {
    float l = std::sqrt(x * x + y * y + z * z + w * w);
    if (l < 1e-6)
      return {0, 0, 0, 0};
    return {x / l, y / l, z / l, w / l};
  }
  operator Vec3() const { return Vec3(x, y, z); }
};
struct Mat4 {
  float m[4][4];
  Mat4() {
    for (int i = 0; i < 4; ++i)
      for (int j = 0; j < 4; ++j)
        m[i][j] = 0.0f;
  }
  static Mat4 identity() {
    Mat4 res;
    for (int i = 0; i < 4; ++i)
      res.m[i][i] = 1.0f;
    return res;
  }

  static Mat4 scale(const Mat4 &m, const Vec3 &s) {
    Mat4 res = Mat4::identity();
    res.m[0][0] = s.x;
    res.m[1][1] = s.y;
    res.m[2][2] = s.z;
    return m.multiply(res);
  }

  static Mat4 translate(const Mat4 &m, const Vec3 &t) {
    Mat4 res = Mat4::identity();
    res.m[0][3] = t.x;
    res.m[1][3] = t.y;
    res.m[2][3] = t.z;
    return m.multiply(res);
  }

  Mat4 multiply(const Mat4 &o) const {
    Mat4 res;
    for (int i = 0; i < 4; ++i)
      for (int j = 0; j < 4; ++j)
        for (int k = 0; k < 4; ++k)
          res.m[i][j] += m[i][k] * o.m[k][j];
    return res;
  }
  Vec4 multiply(const Vec4 &v) const {
    return {m[0][0] * v.x + m[0][1] * v.y + m[0][2] * v.z + m[0][3] * v.w,
            m[1][0] * v.x + m[1][1] * v.y + m[1][2] * v.z + m[1][3] * v.w,
            m[2][0] * v.x + m[2][1] * v.y + m[2][2] * v.z + m[2][3] * v.w,
            m[3][0] * v.x + m[3][1] * v.y + m[3][2] * v.z + m[3][3] * v.w};
  }
  Mat4 operator*(const Mat4 &o) const { return multiply(o); }
  Mat4 transpose() const {
    Mat4 res;
    for (int i = 0; i < 4; ++i)
      for (int j = 0; j < 4; ++j)
        res.m[i][j] = m[j][i];
    return res;
  }
  float determinant3x3(float a, float b, float c, float d, float e, float f,
                       float g, float h, float i) const {
    return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
  }
  float cofactor(int r, int c) const {
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
    return ((r + c) % 2 == 0 ? 1.0f : -1.0f) * det;
  }
  Mat4 inverse() const {
    Mat4 adj;
    for (int i = 0; i < 4; ++i)
      for (int j = 0; j < 4; ++j)
        adj.m[j][i] = cofactor(i, j);
    float det = m[0][0] * adj.m[0][0] + m[0][1] * adj.m[1][0] +
                m[0][2] * adj.m[2][0] + m[0][3] * adj.m[3][0];
    if (std::abs(det) < 1e-6)
      return Mat4::identity();
    Mat4 inv;
    float inv_det = 1.f / det;
    for (int i = 0; i < 4; ++i)
      for (int j = 0; j < 4; ++j)
        inv.m[i][j] = adj.m[i][j] * inv_det;
    return inv;
  }
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
  // static Mat4 perspective(float fov, float aspect, float n, float f) {
  //   Mat4 res = Mat4::identity();
  //   float rad = fov * (std::numbers::pi / 180.f);
  //   float t = std::tan(rad / 2.f);
  //   res.m[0][0] = 1.f / (aspect * t);
  //   res.m[1][1] = 1.f / t;
  //   res.m[2][2] = -(f + n) / (f - n);
  //   res.m[2][3] = -(2 * f * n) / (f - n);
  //   res.m[3][2] = -1.f;
  //   // res.m[3][3] = 0.f;
  //   return res;
  // }
  static Mat4 orthographic(float l, float r, float b, float t, float n,
                           float f) {
    Mat4 res = Mat4::identity();
    res.m[0][0] = 2 / (r - l);
    res.m[1][1] = 2 / (t - b);
    res.m[2][2] = -2 / (f - n);
    res.m[0][3] = -(r + l) / (r - l);
    res.m[1][3] = -(t + b) / (t - b);
    res.m[2][3] = -(f + n) / (f - n);
    return res;
  }
  static Mat4 lookAt(const Vec3 &eye, const Vec3 &target, const Vec3 &up) {
    Vec3 z = (eye - target).normalize();
    Vec3 x = up.cross(z).normalize();
    Vec3 y = z.cross(x).normalize();
    Mat4 res = Mat4::identity();
    res.m[0][0] = x.x;
    res.m[0][1] = x.y;
    res.m[0][2] = x.z;
    res.m[1][0] = y.x;
    res.m[1][1] = y.y;
    res.m[1][2] = y.z;
    res.m[2][0] = z.x;
    res.m[2][1] = z.y;
    res.m[2][2] = z.z;
    res.m[0][3] = -x.dot(eye);
    res.m[1][3] = -y.dot(eye);
    res.m[2][3] = -z.dot(eye);
    return res;
  }
};
Vec3 ndcToScreen(Vec4 ndc, int W, int H) {
  return {(ndc.x * 0.5f + 0.5f) * W, (1.0f - (ndc.y * 0.5f + 0.5f)) * H, ndc.z};
}
float edgeFunc(Vec3 a, Vec3 b, Vec3 c) {
  return (c.x - a.x) * (b.y - a.y) - (c.y - a.y) * (b.x - a.x);
}

float lerp(float a, float b, float t) { return a + (b - a) * t; }

struct Color {
  unsigned char r, g, b;
};
struct Vertex {
  Vec3 pos, normal, tangent;
  Vec2 uv;
};
struct Triangle {
  Vertex v[3];
};
struct Light {
  Vec3 position, direction, color;
  float intensity;
};
struct Texture {
  int w = 0, h = 0;
  std::vector<Color> data;
  Color bilinear(Vec2 uv) const {
    if (w == 0 || h == 0)
      return {0, 0, 0};
    uv.u -= floor(uv.u);
    uv.v -= floor(uv.v);
    float x = uv.u * w - 0.5f, y = uv.v * h - 0.5f;
    int x0 = floor(x), y0 = floor(y);
    float u_f = x - x0, v_f = y - y0;
    auto gc = [&](int xx, int yy) {
      xx = (xx % w + w) % w;
      yy = (yy % h + h) % h;
      return data[yy * w + xx];
    };
    auto lc = [](Color a, Color b, float t) {
      return Color{(unsigned char)(a.r * (1 - t) + b.r * t),
                   (unsigned char)(a.g * (1 - t) + b.g * t),
                   (unsigned char)(a.b * (1 - t) + b.b * t)};
    };
    Color c00 = gc(x0, y0), c10 = gc(x0 + 1, y0), c01 = gc(x0, y0 + 1),
          c11 = gc(x0 + 1, y0 + 1);
    return lc(lc(c00, c10, u_f), lc(c01, c11, u_f), v_f);
  }
};

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
  Vec3 tangent_world;
  Vec2 uv;
  float ndc_z = 0.0f;
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
  Vec3 tangent_world;
  Vec2 uv;
};

// Alpha Blend Shader 输入输出
// struct AlphaBlendShaderIO : public IShaderIO {
//   Vec3 position_world;
//   Vec2 uv;
//   Vec4 position_clip;
//   Vec3 color;
// };

// SSAO Shader 输入输出
struct SSAOShaderIO : public IShaderIO {
  Vec3 position_world;
  Vec3 normal_world;
  Vec2 uv_screen_normalized;
  Vec4 position_clip;
  Vec3 color;
  Vec3 tangent_world;
  Vec2 uv;
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
// struct LightDepthShaderIO : public IShaderIO {
//   Vec3 position_world;
//   float ndc_z = 0.0f;
//   Vec4 position_clip;
//   Vec3 color;
// };

// MultiLight Shader 输入输出
struct MultiLightShaderIO : public IShaderIO {
  Vec3 position_world;
  Vec3 normal_world;
  Vec2 uv;
  Vec4 position_clip;
  Vec3 color;
  Vec3 tangent_world;
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

// struct BloomShaderIO : public IShaderIO {
//   Vec3 position_world;
//   Vec2 uv;
//   Vec4 position_clip;
//   Vec3 color;
//   Vec3 bloom_color; // 新增 bloom_color 用于存储 Bloom 结果
// };

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
  Mat4 ModelMatrix; // 模型矩阵
  Mat4 MVP;             // 模型-视图-投影矩阵
  Mat4 NormalMatrix;    // 法线变换矩阵
  Vec3 eye_pos_world;   // 摄像机世界坐标
  Vec3 light_dir_world; // 世界空间光照方向 (指向光源)
  Texture diffuse_tex;  // 漫反射纹理
  Texture normal_map_tex; // 法线贴图
  // 材质属性
  float ambient_strength, diffuse_strength, specular_strength,
      shininess; // 材质属性

  BlinnPhongShader(const Mat4 &m,const Mat4 &mvp, const Mat4 &nm, const Vec3 &eye,
                   const Vec3 &light_dir, const Texture &tex,const Texture &tex_normal)
      : ModelMatrix(m), MVP(mvp), NormalMatrix(nm), eye_pos_world(eye),
        light_dir_world(light_dir), diffuse_tex(tex), normal_map_tex(tex_normal),ambient_strength(0.1f),
        diffuse_strength(0.2f), specular_strength(0.7f), shininess(32.0f) {}

  void vertex(IShaderIO &io) override {
    auto &data = static_cast<BlinnPhongShaderIO &>(io);
    data.position_clip =
        MVP.multiply({data.position_world.x, data.position_world.y,
                      data.position_world.z, 1.0f});
    data.tangent_world = ModelMatrix.multiply(
        Vec4(data.tangent_world.x, data.tangent_world.y, data.tangent_world.z, 0.0f))
        .normalize();
    data.normal_world = NormalMatrix
        .multiply(Vec4(data.normal_world.x, data.normal_world.y,
                       data.normal_world.z, 0.0f))
        .normalize();
  }

  void fragment(IShaderIO &io) override {
    auto &data = static_cast<BlinnPhongShaderIO &>(io);
    // Vec3 N = NormalMatrix
    //              .multiply(Vec4(data.normal_world.x, data.normal_world.y,
    //                             data.normal_world.z, 0.0f))
    //              .normalize();
    Vec3 N = data.normal_world.normalize();
    Vec3 T = data.tangent_world.normalize();
    //T = (T - N * N.dot(T)).normalize();
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
    // Color c = normal_map_tex.bilinear(data.uv) ;
    // N = {c.r / 255.0f * 2.0f - 1.0f,
    //       c.g / 255.0f * 2.0f - 1.0f,
    //       c.b / 255.0f * 2.0f - 1.0f};
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

struct NormalShader : public IShader {
  Mat4 ModelMatrix; // 模型矩阵
  Mat4 MVP;             // 模型-视图-投影矩阵
  Mat4 NormalMatrix;    // 法线变换矩阵
  Vec3 eye_pos_world;   // 摄像机世界坐标
  Vec3 light_dir_world; // 世界空间光照方向 (指向光源)
  Texture diffuse_tex;  // 漫反射纹理
  Texture normal_map_tex; // 法线贴图
  // 材质属性
  float ambient_strength, diffuse_strength, specular_strength,
      shininess; // 材质属性

  NormalShader(const Mat4 &m,const Mat4 &mvp, const Mat4 &nm, const Vec3 &eye,
                   const Vec3 &light_dir, const Texture &tex,const Texture &tex_normal)
      : ModelMatrix(m), MVP(mvp), NormalMatrix(nm), eye_pos_world(eye),
        light_dir_world(light_dir), diffuse_tex(tex), normal_map_tex(tex_normal),ambient_strength(0.1f),
        diffuse_strength(0.2f), specular_strength(0.7f), shininess(32.0f) {}

  void vertex(IShaderIO &io) override {
    auto &data = static_cast<NormalShaderIO &>(io);
    data.position_clip =
        MVP.multiply({data.position_world.x, data.position_world.y,
                      data.position_world.z, 1.0f});
    data.tangent_world = ModelMatrix.multiply(
        Vec4(data.tangent_world.x, data.tangent_world.y, data.tangent_world.z, 0.0f))
        .normalize();
    data.normal_world = NormalMatrix
        .multiply(Vec4(data.normal_world.x, data.normal_world.y,
                       data.normal_world.z, 0.0f))
        .normalize();
  }

  void fragment(IShaderIO &io) override {
    auto &data = static_cast<NormalShaderIO &>(io);
    // Vec3 N = NormalMatrix
    //              .multiply(Vec4(data.normal_world.x, data.normal_world.y,
    //                             data.normal_world.z, 0.0f))
    //              .normalize();
    Vec3 N = data.normal_world.normalize();
    Vec3 T = data.tangent_world.normalize();
    //T = (T - N * N.dot(T)).normalize();
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
      //N = normal_tangent_space.normalize();
    }
    data.color = {N.x * 0.5f + 0.5f, N.y * 0.5f + 0.5f, N.z * 0.5f + 0.5f};
  }
};

// // Normal Shader (可视化法线)
// struct NormalShader : public IShader {
//   Mat4 MVP;
//   Mat4 NormalMatrix; // 法线变换矩阵
//   NormalShader(const Mat4 &mvp, Mat4 &nm) : MVP(mvp), NormalMatrix(nm) {}

//   void vertex(IShaderIO &io) override {
//     auto &data = static_cast<NormalShaderIO &>(io);
//     data.position_clip =
//         MVP.multiply({data.position_world.x, data.position_world.y,
//                       data.position_world.z, 1.0f});
//   }

//   void fragment(IShaderIO &io) override {
//     auto &data = static_cast<NormalShaderIO &>(io);
//     Vec3 n = NormalMatrix
//                  .multiply(Vec4(data.normal_world.x, data.normal_world.y,
//                                 data.normal_world.z, 0.0f))
//                  .normalize();
//     data.color = {n.x * 0.5f + 0.5f, n.y * 0.5f + 0.5f, n.z * 0.5f + 0.5f};
//   }
// };

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
  Mat4 ModelMatrix; // 模型矩阵
  Vec3 eye_pos_world, light_dir_world;
  Texture diffuse_tex, normal_map_tex;
  float ambient_strength, diffuse_strength, specular_strength, shininess;
  Mat4 NormalMatrix;
  BumpShader(const Mat4& model_matrix ,const Mat4 &mvp, const Mat4 &nm, const Vec3 &eye,
             const Vec3 &light_dir, const Texture &tex,
             const Texture &normalmap)
      : ModelMatrix(model_matrix),MVP(mvp), NormalMatrix(nm), eye_pos_world(eye),
        light_dir_world(light_dir), diffuse_tex(tex), normal_map_tex(normalmap),
        ambient_strength(0.1f), diffuse_strength(0.5f), specular_strength(0.4f),
        shininess(32.0f) {}

  void vertex(IShaderIO &io) override {
    auto &data = static_cast<BumpShaderIO &>(io);
    data.position_clip =
        MVP.multiply({data.position_world.x, data.position_world.y,
                      data.position_world.z, 1.0f});

    data.tangent_world = ModelMatrix.multiply(
        Vec4(data.tangent_world.x, data.tangent_world.y, data.tangent_world.z, 0.0f))
        .normalize();
    data.normal_world = NormalMatrix
        .multiply(Vec4(data.normal_world.x, data.normal_world.y,
                       data.normal_world.z, 0.0f))
        .normalize();
  }

  void fragment(IShaderIO &io) override {
    auto &data = static_cast<BumpShaderIO &>(io);
    // Vec3 N = NormalMatrix
    //              .multiply(Vec4(data.normal_world.x, data.normal_world.y,
    //                             data.normal_world.z, 0.0f))
    //              .normalize();
    Vec3 N = data.normal_world.normalize();
    Vec3 T = data.tangent_world.normalize();
    //T = (T - N * N.dot(T)).normalize();
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
    // Color c = normal_map_tex.bilinear(data.uv) ;
    // N = {c.r / 255.0f * 2.0f - 1.0f,
    //       c.g / 255.0f * 2.0f - 1.0f,
    //       c.b / 255.0f * 2.0f - 1.0f};
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
  Mat4 ModelMatrix;
  Vec3 ambient_light_dir;
  Mat4 NormalMatrix;
  Texture normal_map_tex;
  AOShader(const Mat4& model_matrix ,const Mat4 &mvp, const Mat4 &nm,const Texture &normalmap,
           const Vec3 &amb_dir = Vec3{0.0f, 1.0f, 0.0f})
      : ModelMatrix(model_matrix), MVP(mvp), NormalMatrix(nm), ambient_light_dir(amb_dir.normalize())
      ,normal_map_tex(normalmap) {}

  void vertex(IShaderIO &io) override {
    auto &data = static_cast<AOShaderIO &>(io);
    data.position_clip =
        MVP.multiply({data.position_world.x, data.position_world.y,
                      data.position_world.z, 1.0f});

    data.tangent_world = ModelMatrix.multiply(
        Vec4(data.tangent_world.x, data.tangent_world.y, data.tangent_world.z, 0.0f))
        .normalize();
    data.normal_world = NormalMatrix
        .multiply(Vec4(data.normal_world.x, data.normal_world.y,
                       data.normal_world.z, 0.0f))
        .normalize();
  }

  void fragment(IShaderIO &io) override {
    auto &data = static_cast<AOShaderIO &>(io);

    Vec3 N = data.normal_world.normalize();
    Vec3 T = data.tangent_world.normalize();
    //T = (T - N * N.dot(T)).normalize();
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
    // Vec3 N = data.normal_world.normalize();
    float ao = std::max(0.0f, N.dot(ambient_light_dir));
    ao = std::pow(ao, 0.5f);
    data.color = Vec3{ao, ao, ao};
  }
};

// Alpha Blend Shader (透明混合)
// struct AlphaBlendShader : public IShader {
//   Mat4 MVP;
//   Texture diffuse_tex;
//   float alpha_value;
//   Mat4 NormalMatrix;
//   AlphaBlendShader(const Mat4 &mvp, const Mat4 &nm, const Texture &tex,
//                    float alpha = 0.5f)
//       : MVP(mvp), NormalMatrix(nm), diffuse_tex(tex), alpha_value(alpha) {}

//   void vertex(IShaderIO &io) override {
//     auto &data = static_cast<AlphaBlendShaderIO &>(io);
//     data.position_clip =
//         MVP.multiply({data.position_world.x, data.position_world.y,
//                       data.position_world.z, 1.0f});
//   }

//   void fragment(IShaderIO &io) override {
//     auto &data = static_cast<AlphaBlendShaderIO &>(io);
//     Color tex_color = diffuse_tex.bilinear(data.uv);
//     Vec3 base = {tex_color.r / 255.0f, tex_color.g / 255.0f,
//                  tex_color.b / 255.0f};
//     Vec3 bg = {1.0f, 1.0f, 1.0f};
//     data.color = base * alpha_value + bg * (1.0f - alpha_value);
//   }
// };

// SSAO Shader (改进: 屏幕空间环境光遮蔽模拟)
// 这是一个简化的SSAO，模拟在屏幕空间基于深度缓冲的遮蔽。
// 它会根据像素周围深度值的变化来计算遮蔽，而不是实际的几何体遮挡。
// 需要访问深度缓冲区，但在这个软光栅实现中，我们只能模拟局部深度变化。
struct SSAOShader : public IShader {
  Mat4 MVP;
  Mat4 ViewMatrix;
  Mat4 ModelMatrix;
  float near_plane_z, far_plane_z;
  std::vector<Vec3> ssao_samples;
  Texture noise_texture; // 用于随机化采样方向的噪声纹理
  Texture normal_map_tex; // 法线贴图
  Mat4 NormalMatrix;
  SSAOShader(const Mat4 &model_matrix,const Mat4 &mvp, const Mat4 &nm, const Mat4 &view_mat,
             const Texture& nmap,float near_z, float far_z, int num_samples = 16)
      :ModelMatrix(model_matrix), MVP(mvp), NormalMatrix(nm), ViewMatrix(view_mat), 
      normal_map_tex(nmap),near_plane_z(near_z),far_plane_z(far_z) {
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

    data.tangent_world = ModelMatrix.multiply(
        Vec4(data.tangent_world.x, data.tangent_world.y, data.tangent_world.z, 0.0f))
        .normalize();
    data.normal_world = NormalMatrix
        .multiply(Vec4(data.normal_world.x, data.normal_world.y,
                       data.normal_world.z, 0.0f))
        .normalize();
  }

  void fragment(IShaderIO &io) override {
    auto &data = static_cast<SSAOShaderIO &>(io);
    Vec4 pos_view_4 =
        ViewMatrix.multiply({data.position_world.x, data.position_world.y,
                             data.position_world.z, 1.0f});
    Vec3 pos_view = Vec3(pos_view_4.x, pos_view_4.y, pos_view_4.z);

    Vec3 N = data.normal_world.normalize();
    Vec3 T = data.tangent_world.normalize();
    //T = (T - N * N.dot(T)).normalize();
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

    Vec3 normal_view =
        NormalMatrix
            .multiply(Vec4(N.x, N.y,
                           N.z, 0.0f))
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
// struct LightDepthShader : public IShader {
//   Mat4 LightMVP; // 光源的Model-View-Projection矩阵

//   LightDepthShader(const Mat4 &light_mvp) : LightMVP(light_mvp) {}

//   void vertex(IShaderIO &io) override {
//     auto &data = static_cast<LightDepthShaderIO &>(io);
//     data.position_clip =
//         LightMVP.multiply({data.position_world.x, data.position_world.y,
//                            data.position_world.z, 1.0f});
//   }

//   void fragment(IShaderIO &io) override {
//     auto &data = static_cast<LightDepthShaderIO &>(io);
//     float depth_normalized = (data.ndc_z + 1.0f) * 0.5f;
//     data.color = Vec3{depth_normalized, depth_normalized, depth_normalized};
//   }
// };

// // 光源结构体
// struct Light {
//   Vec3 position;   // 光源位置 (用于点光源)
//   Vec3 direction;  // 光源方向 (从物体指向光源，用于平行光和聚光灯)
//   Vec3 color;      // 光源颜色
//   float intensity; // 光源强度
// };

// MultiLight Blinn-Phong Shader (多光源，无真实阴影)
struct MultiLightBlinnPhongShader : public IShader {
  Mat4 MVP;
  Mat4 ModelMatrix;
  Mat4 NormalMatrix; // 法线矩阵
  Vec3 eye_pos_world;
  std::vector<Light> lights; // 光源列表
  Texture diffuse_tex;
  Texture normal_map_tex; // 法线贴图
  float ambient_strength, diffuse_strength, specular_strength, shininess;

  MultiLightBlinnPhongShader(const Mat4& model_matrix,const Mat4 &mvp, const Mat4& normal_matrix,
                              const Vec3 &eye,
                             const std::vector<Light> &light_list,
                             const Texture &tex,const Texture &normal_map)
      : ModelMatrix(model_matrix),MVP(mvp), eye_pos_world(eye), lights(light_list), 
        diffuse_tex(tex),normal_map_tex(normal_map),NormalMatrix(normal_matrix),
        ambient_strength(0.1f), diffuse_strength(0.7f), specular_strength(0.2f),
        shininess(32.0f) {}
  void vertex(IShaderIO &io) override {
    auto &data = static_cast<MultiLightShaderIO &>(io);
    data.position_clip =
        MVP.multiply({data.position_world.x, data.position_world.y,
                      data.position_world.z, 1.0f});

    data.tangent_world = ModelMatrix.multiply(
        Vec4(data.tangent_world.x, data.tangent_world.y, data.tangent_world.z, 0.0f))
        .normalize();
    data.normal_world = NormalMatrix
        .multiply(Vec4(data.normal_world.x, data.normal_world.y,
                       data.normal_world.z, 0.0f))
        .normalize();
  }

  void fragment(IShaderIO &io) override {
    auto &data = static_cast<MultiLightShaderIO &>(io);
    // Vec3 N = data.normal_world.normalize();
    Vec3 N = data.normal_world.normalize();
    Vec3 T = data.tangent_world.normalize();
    //T = (T - N * N.dot(T)).normalize();
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

// struct BloomShader : public IShader {
//   Mat4 MVP;
//   Texture input_texture; // 输入纹理
//   float bloom_strength;  // Bloom强度

//   BloomShader(const Mat4 &mvp, const Texture &tex, float strength = 1.0f)
//       : MVP(mvp), input_texture(tex), bloom_strength(strength) {}

//   void vertex(IShaderIO &io) override {
//     auto &data = static_cast<BloomShaderIO &>(io);
//     data.position_clip =
//         MVP.multiply({data.position_world.x, data.position_world.y,
//                       data.position_world.z, 1.0f});
//   }

//   void fragment(IShaderIO &io) override {
//     auto &data = static_cast<BloomShaderIO &>(io);
//     Color tex_color = input_texture.bilinear(data.uv);
//     Vec3 color = {tex_color.r / 255.0f, tex_color.g / 255.0f,
//                   tex_color.b / 255.0f};
//     float lum = 0.2126f * color.x + 0.7152f * color.y + 0.0722f * color.z;
//     if (lum > 0.7f) { // 阈值可调
//       data.bloom_color = color * bloom_strength;
//     } else {
//       data.bloom_color = Vec3{0.0f, 0.0f, 0.0f};
//     }
//   }
// };

template <typename T>
T baryInterp(const T &v0, const T &v1, const T &v2, float w0, float w1,
             float w2) {
  return v0 * w0 + v1 * w1 + v2 * w2;
}
const int MSAA_SAMPLES_1 = 1;
const float MSAA_OFFSETS_1[1][2] = {{0.5f, 0.5f}};
const int MSAA_SAMPLES_4 = 4;
const float MSAA_OFFSETS_4[4][2] = {
    {.375f, .125f}, {.875f, .375f}, {.125f, .625f}, {.625f, .875f}};

struct Mesh {
  std::vector<Triangle> triangles;
};
Mesh loadModel(const std::string &f) {
  Mesh m;
  tinyobj::attrib_t a;
  std::vector<tinyobj::shape_t> s;
  std::vector<tinyobj::material_t> mt;
  std::string w, e;
  if (!tinyobj::LoadObj(&a, &s, &mt, &w, &e, f.c_str())) {
    std::cerr << "ERR:" << e << std::endl;
    return m;
  }
  for (const auto &sh : s) {
    for (size_t i = 0; i < sh.mesh.indices.size() / 3; i++) {
      Triangle t;
      for (int v = 0; v < 3; v++) {
        tinyobj::index_t idx = sh.mesh.indices[3 * i + v];
        t.v[v].pos = {a.vertices[3 * idx.vertex_index + 0],
                      a.vertices[3 * idx.vertex_index + 1],
                      a.vertices[3 * idx.vertex_index + 2]};
        if (!a.normals.empty() && idx.normal_index >= 0)
          t.v[v].normal = {a.normals[3 * idx.normal_index + 0],
                           a.normals[3 * idx.normal_index + 1],
                           a.normals[3 * idx.normal_index + 2]};

        if (a.normals.empty())
          t.v[v].normal = {0, 0, 1}; // 默认法线
        
        if (!a.texcoords.empty() && idx.texcoord_index >= 0)
          t.v[v].uv = {a.texcoords[2 * idx.texcoord_index + 0],
                       a.texcoords[2 * idx.texcoord_index + 1]};
      }
      Vec3 e1 = t.v[1].pos - t.v[0].pos, e2 = t.v[2].pos - t.v[0].pos;
      Vec2 d1 = t.v[1].uv - t.v[0].uv, d2 = t.v[2].uv - t.v[0].uv;
      float det = d1.u * d2.v - d2.u * d1.v;
      float inv_det = std::abs(det) < 1e-6 ? 1.f : 1.f / det;
      Vec3 tan = (e1 * d2.v - e2 * d1.v) * inv_det;
      for (int v = 0; v < 3; v++)
        t.v[v].tangent = tan.normalize();
      m.triangles.push_back(t);
    }
  }
  return m;
}
Texture loadTexture(const std::string &f) {
  Texture t;
  int n;
  stbi_set_flip_vertically_on_load(true);
  unsigned char *d = stbi_load(f.c_str(), &t.w, &t.h, &n, 3);
  if (!d) {
    std::cerr << "Failed to load texture:" << f << std::endl;
    return t;
  }
  t.data.resize(t.w * t.h);
  for (int i = 0; i < t.w * t.h; ++i)
    t.data[i] = {d[3 * i + 0], d[3 * i + 1], d[3 * i + 2]};
  stbi_image_free(d);
  return t;
}

class Rasterizer {
public:
  Rasterizer(int w, int h, int msaa, const float (*offs)[2])
      : W(w), H(h), msaaSamples(msaa), msaaOffsets(offs) {
    clearBuffers();
  }
  void clearBuffers() {
    color_buffer.assign(W * H * msaaSamples, {0, 0, 0});
    z_buffer.assign(W * H * msaaSamples, std::numeric_limits<float>::max());
  }
  void rasterizeTriangle(
      const Vec3 s[3], const float inv_w[3], const Vec4 c[3],
      const std::function<Vec3(float, float, float, float, float)> &getFrag) {
    float area = edgeFunc(s[0], s[1], s[2]);
    if (area <= 1e-5)
      return;
    int min_x = std::max(0, (int)floor(std::min({s[0].x, s[1].x, s[2].x})));
    int max_x = std::min(W - 1, (int)ceil(std::max({s[0].x, s[1].x, s[2].x})));
    int min_y = std::max(0, (int)floor(std::min({s[0].y, s[1].y, s[2].y})));
    int max_y = std::min(H - 1, (int)ceil(std::max({s[0].y, s[1].y, s[2].y})));
    for (int y = min_y; y <= max_y; ++y)
      for (int x = min_x; x <= max_x; ++x)
        for (int i = 0; i < msaaSamples; ++i) {
          Vec3 p = {x + msaaOffsets[i][0], y + msaaOffsets[i][1], 0};
          float w0 = edgeFunc(s[1], s[2], p), w1 = edgeFunc(s[2], s[0], p),
                w2 = edgeFunc(s[0], s[1], p);
          if (w0 >= 0 && w1 >= 0 && w2 >= 0) {

            float inv_a = 1.f / area;
            w0 *= inv_a;
            w1 *= inv_a;
            w2 *= inv_a;
            float inv_w_interp =
                baryInterp(inv_w[0], inv_w[1], inv_w[2], w0, w1, w2);
            float w_interp = 1.f / inv_w_interp;
            float z_ndc_0 = c[0].z * inv_w[0];
            float z_ndc_1 = c[1].z * inv_w[1];
            float z_ndc_2 = c[2].z * inv_w[2];
            float z_ndc = baryInterp(z_ndc_0, z_ndc_1, z_ndc_2, w0, w1, w2);
            int idx = (y * W + x) * msaaSamples + i;
            if (z_ndc < z_buffer[idx]) {
              z_buffer[idx] = z_ndc;
              color_buffer[idx] = getFrag(w0, w1, w2, w_interp, z_ndc);
            }
          }
        }
  }
  void writeToFile(const std::string &file) const {
    std::string dir = "output_images/";
    if (!std::filesystem::exists(dir))
      std::filesystem::create_directories(dir);
    std::ofstream ofs(dir + file);
    if (!ofs.is_open())
      return;
    ofs << "P3\n" << W << " " << H << "\n255\n";
    for (int y = 0; y < H; ++y)
      for (int x = 0; x < W; ++x) {
        Vec3 avg_c;
        for (int s = 0; s < msaaSamples; ++s)
          avg_c = avg_c + color_buffer[(y * W + x) * msaaSamples + s];
        avg_c = avg_c * (1.f / msaaSamples);
        ofs << std::clamp((int)(avg_c.x * 255.99f), 0, 255) << " "
            << std::clamp((int)(avg_c.y * 255.99f), 0, 255) << " "
            << std::clamp((int)(avg_c.z * 255.99f), 0, 255) << "\n";
      }
    std::cout << "Done: " << file << std::endl;
  }

private:
  int W, H, msaaSamples;
  const float (*msaaOffsets)[2];
  std::vector<Vec3> color_buffer;
  std::vector<float> z_buffer;
};

class IDrawPass {
public:
  virtual ~IDrawPass() = default;
  virtual void execute(int, int) = 0;
  virtual const std::string &getName() const = 0;
};
class DrawGraph {
public:
  void addPass(std::unique_ptr<IDrawPass> p) { passes.push_back(std::move(p)); }
  void render(int W, int H) {
    for (auto &p : passes)
      p->execute(W, H);
  }

private:
  std::vector<std::unique_ptr<IDrawPass>> passes;
};

// 宏定义以简化具体Pass类的创建
#define DRAW_PASS_BOILERPLATE(ClassName, ShaderName, ShaderIO)                 \
public:                                                                        \
  const std::string &getName() const override { return passName; }             \
  void execute(int W, int H) override {                                        \
    Rasterizer r(W, H, msaaSamples, msaaOffsets);                              \
    for (const auto &tri : mesh->triangles) {                                  \
      ShaderIO v_out[3];                                                       \
      Vec4 c[3];                                                               \
      float inv_w[3];                                                          \
      Vec3 s[3];                                                               \
      for (int i = 0; i < 3; ++i) {                                            \
        initializeVertex(v_out[i], tri.v[i]);                                  \
        shader->vertex(v_out[i]);                                              \
        c[i] = v_out[i].position_clip;                                         \
        inv_w[i] = 1.0f / c[i].w;                                              \
        s[i] = ndcToScreen(c[i] * inv_w[i], W, H);                             \
      }                                                                        \
      auto getFrag = [&](float w0, float w1, float w2, float iw,               \
                         float nz) -> Vec3 {                                   \
        static ShaderIO f_in;                                                  \
        interpolateFragment(f_in, v_out, w0, w1, w2, iw, nz, s, W, H);         \
        shader->fragment(f_in);                                                \
        return getOutputColor(f_in);                                           \
      };                                                                       \
      r.rasterizeTriangle(s, inv_w, c, getFrag);                               \
    }                                                                          \
    r.writeToFile(passName);                                                   \
  }                                                                            \
                                                                               \
private:                                                                       \
  std::string passName;                                                        \
  Mesh *mesh;                                                                  \
  std::unique_ptr<ShaderName> shader;                                          \
  int msaaSamples;                                                             \
  const float(*msaaOffsets)[2];

// ========================
// 16个具体DrawPass实现
// ========================

// 1. BlinnPhong
class BlinnPhongPass : public IDrawPass {
  DRAW_PASS_BOILERPLATE(BlinnPhongPass, BlinnPhongShader, BlinnPhongShaderIO)
public:
  BlinnPhongPass(std::string n,Mat4 model_matrix ,Mat4 mvp, Mat4 nm, Vec3 eye, Vec3 l,
                 const Texture &t, const Texture &tn,Mesh *m, int ms = MSAA_SAMPLES_4,
                 const float (*off)[2] = MSAA_OFFSETS_4)
      : passName(n), mesh(m), msaaSamples(ms), msaaOffsets(off) {
    shader = std::make_unique<BlinnPhongShader>(model_matrix,mvp, nm, eye, l, t,tn);
  }

private:
  void initializeVertex(BlinnPhongShaderIO &io, const Vertex &v) {
    io.position_world = v.pos;
    io.normal_world = v.normal;
    io.tangent_world = v.tangent;
    io.uv = v.uv;
  }
  void interpolateFragment(BlinnPhongShaderIO &f, const BlinnPhongShaderIO v[3],
                           float w0, float w1, float w2, float iw, float nz,
                           const Vec3 s[3], int W, int H) {
    f.position_world =
        baryInterp(v[0].position_world * (1 / v[0].position_clip.w),
                   v[1].position_world * (1 / v[1].position_clip.w),
                   v[2].position_world * (1 / v[2].position_clip.w), w0, w1,
                   w2) *
        iw;
    f.normal_world =
        baryInterp(v[0].normal_world * (1 / v[0].position_clip.w),
                   v[1].normal_world * (1 / v[1].position_clip.w),
                   v[2].normal_world * (1 / v[2].position_clip.w), w0, w1, w2) *
        iw;
    f.tangent_world =
        baryInterp(v[0].tangent_world * (1 / v[0].position_clip.w),
                   v[1].tangent_world * (1 / v[1].position_clip.w),
                   v[2].tangent_world * (1 / v[2].position_clip.w), w0, w1,
                   w2) *
        iw;
    f.uv = baryInterp(v[0].uv * (1 / v[0].position_clip.w),
                      v[1].uv * (1 / v[1].position_clip.w),
                      v[2].uv * (1 / v[2].position_clip.w), w0, w1, w2) *
           iw;
    f.ndc_z = nz;
  }
  Vec3 getOutputColor(const BlinnPhongShaderIO &f) { return f.color; }
};

// 2. Normal
class NormalPass : public IDrawPass {
  DRAW_PASS_BOILERPLATE(NormalPass, NormalShader, NormalShaderIO)
public:
  NormalPass(std::string n,Mat4 model_matrix ,Mat4 mvp, Mat4 nm, Vec3 eye, Vec3 l,
                 const Texture &t, const Texture &tn,Mesh *m, int ms = MSAA_SAMPLES_4,
                 const float (*off)[2] = MSAA_OFFSETS_4)
      : passName(n), mesh(m), msaaSamples(ms), msaaOffsets(off) {
    shader = std::make_unique<NormalShader>(model_matrix,mvp, nm, eye, l, t,tn);
  }

private:
  void initializeVertex(NormalShaderIO &io, const Vertex &v) {
    io.position_world = v.pos;
    io.normal_world = v.normal;
    io.tangent_world = v.tangent;
    io.uv = v.uv;
  }
  void interpolateFragment(NormalShaderIO &f, const NormalShaderIO v[3],
                           float w0, float w1, float w2, float iw, float nz,
                           const Vec3 s[3], int W, int H) {
    f.position_world =
        baryInterp(v[0].position_world * (1 / v[0].position_clip.w),
                   v[1].position_world * (1 / v[1].position_clip.w),
                   v[2].position_world * (1 / v[2].position_clip.w), w0, w1,
                   w2) *
        iw;
    f.normal_world =
        baryInterp(v[0].normal_world * (1 / v[0].position_clip.w),
                   v[1].normal_world * (1 / v[1].position_clip.w),
                   v[2].normal_world * (1 / v[2].position_clip.w), w0, w1, w2) *
        iw;
    f.tangent_world =
        baryInterp(v[0].tangent_world * (1 / v[0].position_clip.w),
                   v[1].tangent_world * (1 / v[1].position_clip.w),
                   v[2].tangent_world * (1 / v[2].position_clip.w), w0, w1,
                   w2) *
        iw;
    f.uv = baryInterp(v[0].uv * (1 / v[0].position_clip.w),
                      v[1].uv * (1 / v[1].position_clip.w),
                      v[2].uv * (1 / v[2].position_clip.w), w0, w1, w2) *
           iw;
    f.ndc_z = nz;
  }
  Vec3 getOutputColor(const NormalShaderIO &f) { return f.color; }
};

// 3. Texture
class TexturePass : public IDrawPass {
  DRAW_PASS_BOILERPLATE(TexturePass, TextureShader, TextureShaderIO)
public:
  TexturePass(std::string n, Mat4 mvp, Mat4 nm, const Texture &t, Mesh *m,
              int ms = MSAA_SAMPLES_4, const float (*off)[2] = MSAA_OFFSETS_4)
      : passName(n), mesh(m), msaaSamples(ms), msaaOffsets(off) {
    shader = std::make_unique<TextureShader>(mvp, nm, t);
  }

private:
  void initializeVertex(TextureShaderIO &io, const Vertex &v) {
    io.position_world = v.pos;
    io.uv = v.uv;
  }
  void interpolateFragment(TextureShaderIO &f, const TextureShaderIO v[3],
                           float w0, float w1, float w2, float iw, float nz,
                           const Vec3 s[3], int W, int H) {
    f.position_world =
        baryInterp(v[0].position_world * (1 / v[0].position_clip.w),
                   v[1].position_world * (1 / v[1].position_clip.w),
                   v[2].position_world * (1 / v[2].position_clip.w), w0, w1,
                   w2) *
        iw;
    f.uv = baryInterp(v[0].uv * (1 / v[0].position_clip.w),
                      v[1].uv * (1 / v[1].position_clip.w),
                      v[2].uv * (1 / v[2].position_clip.w), w0, w1, w2) *
           iw;
  }
  Vec3 getOutputColor(const TextureShaderIO &f) { return f.color; }
};

// 4. Bump
class BumpPass : public IDrawPass {
  DRAW_PASS_BOILERPLATE(BumpPass, BumpShader, BumpShaderIO)
public:
  BumpPass(std::string n,Mat4 model_matrix, Mat4 mvp, Mat4 nm, Vec3 eye, Vec3 l, const Texture &t,
           const Texture &nmap, Mesh *m, int ms = MSAA_SAMPLES_4,
           const float (*off)[2] = MSAA_OFFSETS_4)
      : passName(n), mesh(m), msaaSamples(ms), msaaOffsets(off) {
    shader = std::make_unique<BumpShader>(model_matrix,mvp, nm, eye, l, t, nmap);
  }

private:
  void initializeVertex(BumpShaderIO &io, const Vertex &v) {
    io.position_world = v.pos;
    io.normal_world = v.normal;
    io.tangent_world = v.tangent;
    io.uv = v.uv;
  }
  void interpolateFragment(BumpShaderIO &f, const BumpShaderIO v[3], float w0,
                           float w1, float w2, float iw, float nz,
                           const Vec3 s[3], int W, int H) {
    f.position_world =
        baryInterp(v[0].position_world * (1 / v[0].position_clip.w),
                   v[1].position_world * (1 / v[1].position_clip.w),
                   v[2].position_world * (1 / v[2].position_clip.w), w0, w1,
                   w2) *
        iw;
    f.normal_world =
        baryInterp(v[0].normal_world * (1 / v[0].position_clip.w),
                   v[1].normal_world * (1 / v[1].position_clip.w),
                   v[2].normal_world * (1 / v[2].position_clip.w), w0, w1, w2) *
        iw;
    f.tangent_world =
        baryInterp(v[0].tangent_world * (1 / v[0].position_clip.w),
                   v[1].tangent_world * (1 / v[1].position_clip.w),
                   v[2].tangent_world * (1 / v[2].position_clip.w), w0, w1,
                   w2) *
        iw;
    f.uv = baryInterp(v[0].uv * (1 / v[0].position_clip.w),
                      v[1].uv * (1 / v[1].position_clip.w),
                      v[2].uv * (1 / v[2].position_clip.w), w0, w1, w2) *
           iw;
  }
  Vec3 getOutputColor(const BumpShaderIO &f) { return f.color; }
};

// // 5. Displacement
// class DisplacementPass : public IDrawPass {
//   DRAW_PASS_BOILERPLATE(DisplacementPass, DisplacementShader,
//                         DisplacementShaderIO)
// public:
//   DisplacementPass(std::string n, Mat4 mvp, Mat4 nm, Vec3 eye, Vec3 l,
//                    const Texture &t, const Texture &dmap, float scale, Mesh *m,
//                    int ms = MSAA_SAMPLES_4,
//                    const float (*off)[2] = MSAA_OFFSETS_4)
//       : passName(n), mesh(m), msaaSamples(ms), msaaOffsets(off) {
//     shader =
//         std::make_unique<DisplacementShader>(mvp, nm, eye, l, t, dmap, scale);
//   }

// private:
//   void initializeVertex(DisplacementShaderIO &io, const Vertex &v) {
//     io.position_world = v.pos;
//     io.normal_world = v.normal;
//     io.tangent_world = v.tangent;
//     io.uv = v.uv;
//   }
//   void interpolateFragment(DisplacementShaderIO &f,
//                            const DisplacementShaderIO v[3], float w0, float w1,
//                            float w2, float iw, float nz, const Vec3 s[3], int W,
//                            int H) {
//     f.position_world =
//         baryInterp(v[0].position_world * (1 / v[0].position_clip.w),
//                    v[1].position_world * (1 / v[1].position_clip.w),
//                    v[2].position_world * (1 / v[2].position_clip.w), w0, w1,
//                    w2) *
//         iw;
//     f.normal_world =
//         baryInterp(v[0].normal_world * (1 / v[0].position_clip.w),
//                    v[1].normal_world * (1 / v[1].position_clip.w),
//                    v[2].normal_world * (1 / v[2].position_clip.w), w0, w1, w2) *
//         iw;
//     f.tangent_world =
//         baryInterp(v[0].tangent_world * (1 / v[0].position_clip.w),
//                    v[1].tangent_world * (1 / v[1].position_clip.w),
//                    v[2].tangent_world * (1 / v[2].position_clip.w), w0, w1,
//                    w2) *
//         iw;
//     f.uv = baryInterp(v[0].uv * (1 / v[0].position_clip.w),
//                       v[1].uv * (1 / v[1].position_clip.w),
//                       v[2].uv * (1 / v[2].position_clip.w), w0, w1, w2) *
//            iw;
//   }
//   Vec3 getOutputColor(const DisplacementShaderIO &f) { return f.color; }
// };

// 6. Parallax
class ParallaxPass : public IDrawPass {
  DRAW_PASS_BOILERPLATE(ParallaxPass, ParallaxShader, ParallaxShaderIO)
public:
  ParallaxPass(std::string n, Mat4 mvp, Mat4 nm, Vec3 eye, Vec3 l,
               const Texture &t, const Texture &pmap, const Texture &nmap,
               float scale, Mesh *m, int ms = MSAA_SAMPLES_4,
               const float (*off)[2] = MSAA_OFFSETS_4)
      : passName(n), mesh(m), msaaSamples(ms), msaaOffsets(off) {
    shader =
        std::make_unique<ParallaxShader>(mvp, nm, eye, l, t, pmap, nmap, scale);
  }

private:
  void initializeVertex(ParallaxShaderIO &io, const Vertex &v) {
    io.position_world = v.pos;
    io.normal_world = v.normal;
    io.tangent_world = v.tangent;
    io.uv = v.uv;
  }
  void interpolateFragment(ParallaxShaderIO &f, const ParallaxShaderIO v[3],
                           float w0, float w1, float w2, float iw, float nz,
                           const Vec3 s[3], int W, int H) {
    f.position_world =
        baryInterp(v[0].position_world * (1 / v[0].position_clip.w),
                   v[1].position_world * (1 / v[1].position_clip.w),
                   v[2].position_world * (1 / v[2].position_clip.w), w0, w1,
                   w2) *
        iw;
    f.normal_world =
        baryInterp(v[0].normal_world * (1 / v[0].position_clip.w),
                   v[1].normal_world * (1 / v[1].position_clip.w),
                   v[2].normal_world * (1 / v[2].position_clip.w), w0, w1, w2) *
        iw;
    f.tangent_world =
        baryInterp(v[0].tangent_world * (1 / v[0].position_clip.w),
                   v[1].tangent_world * (1 / v[1].position_clip.w),
                   v[2].tangent_world * (1 / v[2].position_clip.w), w0, w1,
                   w2) *
        iw;
    f.uv = baryInterp(v[0].uv * (1 / v[0].position_clip.w),
                      v[1].uv * (1 / v[1].position_clip.w),
                      v[2].uv * (1 / v[2].position_clip.w), w0, w1, w2) *
           iw;
  }
  Vec3 getOutputColor(const ParallaxShaderIO &f) { return f.color; }
};

// 7. Shadow
// class ShadowPass : public IDrawPass {
//   DRAW_PASS_BOILERPLATE(ShadowPass, ShadowShader, ShadowShaderIO)
// public:
//   ShadowPass(std::string n, Mat4 mvp, Mat4 nm, Vec3 l, Mesh *m,
//              int ms = MSAA_SAMPLES_4, const float (*off)[2] = MSAA_OFFSETS_4)
//       : passName(n), mesh(m), msaaSamples(ms), msaaOffsets(off) {
//     shader = std::make_unique<ShadowShader>(mvp, nm, l);
//   }

// private:
//   void initializeVertex(ShadowShaderIO &io, const Vertex &v) {
//     io.position_world = v.pos;
//     io.normal_world = v.normal;
//   }
//   void interpolateFragment(ShadowShaderIO &f, const ShadowShaderIO v[3],
//                            float w0, float w1, float w2, float iw, float nz,
//                            const Vec3 s[3], int W, int H) {
//     f.position_world =
//         baryInterp(v[0].position_world * (1 / v[0].position_clip.w),
//                    v[1].position_world * (1 / v[1].position_clip.w),
//                    v[2].position_world * (1 / v[2].position_clip.w), w0, w1,
//                    w2) *
//         iw;
//     f.normal_world =
//         baryInterp(v[0].normal_world * (1 / v[0].position_clip.w),
//                    v[1].normal_world * (1 / v[1].position_clip.w),
//                    v[2].normal_world * (1 / v[2].position_clip.w), w0, w1, w2) *
//         iw;
//   }
//   Vec3 getOutputColor(const ShadowShaderIO &f) { return f.color; }
// };

// 8. AO
class AOPass : public IDrawPass {
  DRAW_PASS_BOILERPLATE(AOPass, AOShader, AOShaderIO)
public:
  AOPass(std::string n,Mat4 model_matrix, Mat4 mvp, Mat4 nm, Vec3 amb_dir, Mesh *m,
         const Texture& nmap,int ms = MSAA_SAMPLES_4, const float (*off)[2] = MSAA_OFFSETS_4)
      : passName(n), mesh(m), msaaSamples(ms), msaaOffsets(off) {
    shader = std::make_unique<AOShader>(model_matrix,mvp, nm, nmap,amb_dir);
  }

private:
  void initializeVertex(AOShaderIO &io, const Vertex &v) {
    io.position_world = v.pos;
    io.normal_world = v.normal;
  }
  void interpolateFragment(AOShaderIO &f, const AOShaderIO v[3], float w0,
                           float w1, float w2, float iw, float nz,
                           const Vec3 s[3], int W, int H) {
    f.position_world =
        baryInterp(v[0].position_world * (1 / v[0].position_clip.w),
                   v[1].position_world * (1 / v[1].position_clip.w),
                   v[2].position_world * (1 / v[2].position_clip.w), w0, w1,
                   w2) *
        iw;
    f.normal_world =
        baryInterp(v[0].normal_world * (1 / v[0].position_clip.w),
                   v[1].normal_world * (1 / v[1].position_clip.w),
                   v[2].normal_world * (1 / v[2].position_clip.w), w0, w1, w2) *
        iw;
    f.tangent_world =
        baryInterp(v[0].tangent_world * (1 / v[0].position_clip.w),
                   v[1].tangent_world * (1 / v[1].position_clip.w),
                   v[2].tangent_world * (1 / v[2].position_clip.w), w0, w1,
                   w2) *
        iw;
    f.uv = baryInterp(v[0].uv * (1 / v[0].position_clip.w),
                      v[1].uv * (1 / v[1].position_clip.w),
                      v[2].uv * (1 / v[2].position_clip.w), w0, w1, w2) *
           iw;
  }
  Vec3 getOutputColor(const AOShaderIO &f) { return f.color; }
};

// // 9. AlphaBlend
// class AlphaBlendPass : public IDrawPass {
//   DRAW_PASS_BOILERPLATE(AlphaBlendPass, AlphaBlendShader, AlphaBlendShaderIO)
// public:
//   AlphaBlendPass(std::string n, Mat4 mvp, Mat4 nm, const Texture &t,
//                  float alpha, Mesh *m, int ms = MSAA_SAMPLES_4,
//                  const float (*off)[2] = MSAA_OFFSETS_4)
//       : passName(n), mesh(m), msaaSamples(ms), msaaOffsets(off) {
//     shader = std::make_unique<AlphaBlendShader>(mvp, nm, t, alpha);
//   }

// private:
//   void initializeVertex(AlphaBlendShaderIO &io, const Vertex &v) {
//     io.position_world = v.pos;
//     io.uv = v.uv;
//   }
//   void interpolateFragment(AlphaBlendShaderIO &f, const AlphaBlendShaderIO v[3],
//                            float w0, float w1, float w2, float iw, float nz,
//                            const Vec3 s[3], int W, int H) {
//     f.position_world =
//         baryInterp(v[0].position_world * (1 / v[0].position_clip.w),
//                    v[1].position_world * (1 / v[1].position_clip.w),
//                    v[2].position_world * (1 / v[2].position_clip.w), w0, w1,
//                    w2) *
//         iw;
//     f.uv = baryInterp(v[0].uv * (1 / v[0].position_clip.w),
//                       v[1].uv * (1 / v[1].position_clip.w),
//                       v[2].uv * (1 / v[2].position_clip.w), w0, w1, w2) *
//            iw;
//   }
//   Vec3 getOutputColor(const AlphaBlendShaderIO &f) { return f.color; }
// };

// 10. SSAO
class SSAOPass : public IDrawPass {
  DRAW_PASS_BOILERPLATE(SSAOPass, SSAOShader, SSAOShaderIO)
public:
  SSAOPass(std::string n,Mat4 model_matrix, Mat4 mvp, Mat4 nm, Mat4 view, float nearz, float farz,
           Mesh *m, const Texture& nmap,int ms = MSAA_SAMPLES_4,
           const float (*off)[2] = MSAA_OFFSETS_4)
      : passName(n), mesh(m), msaaSamples(ms), msaaOffsets(off) {
    shader = std::make_unique<SSAOShader>(model_matrix,mvp, nm, view, nmap,nearz, farz, 16);
  }

private:
  void initializeVertex(SSAOShaderIO &io, const Vertex &v) {
    io.position_world = v.pos;
    io.normal_world = v.normal;
  }
  void interpolateFragment(SSAOShaderIO &f, const SSAOShaderIO v[3], float w0,
                           float w1, float w2, float iw, float nz,
                           const Vec3 s[3], int W, int H) {
    f.position_world =
        baryInterp(v[0].position_world * (1 / v[0].position_clip.w),
                   v[1].position_world * (1 / v[1].position_clip.w),
                   v[2].position_world * (1 / v[2].position_clip.w), w0, w1,
                   w2) *
        iw;
    f.normal_world =
        baryInterp(v[0].normal_world * (1 / v[0].position_clip.w),
                   v[1].normal_world * (1 / v[1].position_clip.w),
                   v[2].normal_world * (1 / v[2].position_clip.w), w0, w1, w2) *
        iw;
    f.uv_screen_normalized =
        baryInterp(Vec2{s[0].x / W, s[0].y / H} * (1 / v[0].position_clip.w),
                   Vec2{s[1].x / W, s[1].y / H} * (1 / v[1].position_clip.w),
                   Vec2{s[2].x / W, s[2].y / H} * (1 / v[2].position_clip.w),
                   w0, w1, w2) *
        iw;

    f.tangent_world =
        baryInterp(v[0].tangent_world * (1 / v[0].position_clip.w),
                   v[1].tangent_world * (1 / v[1].position_clip.w),
                   v[2].tangent_world * (1 / v[2].position_clip.w), w0, w1,
                   w2) *
        iw;
    f.uv = baryInterp(v[0].uv * (1 / v[0].position_clip.w),
                      v[1].uv * (1 / v[1].position_clip.w),
                      v[2].uv * (1 / v[2].position_clip.w), w0, w1, w2) *
           iw;
  }
  Vec3 getOutputColor(const SSAOShaderIO &f) { return f.color; }
};

// 11. SpotLight
class SpotLightPass : public IDrawPass {
  DRAW_PASS_BOILERPLATE(SpotLightPass, SpotLightShader, SpotLightShaderIO)
public:
  SpotLightPass(std::string n, Mat4 mvp, Vec3 eye, Vec3 light_pos,
                Vec3 spot_dir, const Texture &t, float inner_angle,
                float outer_angle, Mesh *m, int ms = MSAA_SAMPLES_4,
                const float (*off)[2] = MSAA_OFFSETS_4)
      : passName(n), mesh(m), msaaSamples(ms), msaaOffsets(off) {
    shader = std::make_unique<SpotLightShader>(mvp, eye, light_pos, spot_dir, t,
                                               inner_angle, outer_angle);
  }

private:
  void initializeVertex(SpotLightShaderIO &io, const Vertex &v) {
    io.position_world = v.pos;
    io.normal_world = v.normal;
    io.uv = v.uv;
  }
  void interpolateFragment(SpotLightShaderIO &f, const SpotLightShaderIO v[3],
                           float w0, float w1, float w2, float iw, float nz,
                           const Vec3 s[3], int W, int H) {
    f.position_world =
        baryInterp(v[0].position_world * (1 / v[0].position_clip.w),
                   v[1].position_world * (1 / v[1].position_clip.w),
                   v[2].position_world * (1 / v[2].position_clip.w), w0, w1,
                   w2) *
        iw;
    f.normal_world =
        baryInterp(v[0].normal_world * (1 / v[0].position_clip.w),
                   v[1].normal_world * (1 / v[1].position_clip.w),
                   v[2].normal_world * (1 / v[2].position_clip.w), w0, w1, w2) *
        iw;
    f.uv = baryInterp(v[0].uv * (1 / v[0].position_clip.w),
                      v[1].uv * (1 / v[1].position_clip.w),
                      v[2].uv * (1 / v[2].position_clip.w), w0, w1, w2) *
           iw;
  }
  Vec3 getOutputColor(const SpotLightShaderIO &f) { return f.color; }
};

// 12. Depth
class DepthPass : public IDrawPass {
  DRAW_PASS_BOILERPLATE(DepthPass, DepthShader, DepthShaderIO)
public:
  DepthPass(std::string n, Mat4 mvp, Mesh *m, int ms = MSAA_SAMPLES_4,
            const float (*off)[2] = MSAA_OFFSETS_4)
      : passName(n), mesh(m), msaaSamples(ms), msaaOffsets(off) {
    shader = std::make_unique<DepthShader>(mvp);
  }

private:
  void initializeVertex(DepthShaderIO &io, const Vertex &v) {
    io.position_world = v.pos;
  }
  void interpolateFragment(DepthShaderIO &f, const DepthShaderIO v[3], float w0,
                           float w1, float w2, float iw, float nz,
                           const Vec3 s[3], int W, int H) {
    f.position_world =
        baryInterp(v[0].position_world * (1 / v[0].position_clip.w),
                   v[1].position_world * (1 / v[1].position_clip.w),
                   v[2].position_world * (1 / v[2].position_clip.w), w0, w1,
                   w2) *
        iw;
    f.ndc_z = nz;
  }
  Vec3 getOutputColor(const DepthShaderIO &f) { return f.color; }
};

// // 13. LightDepth
// class LightDepthPass : public IDrawPass {
//   DRAW_PASS_BOILERPLATE(LightDepthPass, LightDepthShader, LightDepthShaderIO)
// public:
//   LightDepthPass(std::string n, Mat4 mvp, Mesh *m, int ms = MSAA_SAMPLES_4,
//                  const float (*off)[2] = MSAA_OFFSETS_4)
//       : passName(n), mesh(m), msaaSamples(ms), msaaOffsets(off) {
//     shader = std::make_unique<LightDepthShader>(mvp);
//   }

// private:
//   void initializeVertex(LightDepthShaderIO &io, const Vertex &v) {
//     io.position_world = v.pos;
//   }
//   void interpolateFragment(LightDepthShaderIO &f, const LightDepthShaderIO v[3],
//                            float w0, float w1, float w2, float iw, float nz,
//                            const Vec3 s[3], int W, int H) {
//     f.position_world =
//         baryInterp(v[0].position_world * (1 / v[0].position_clip.w),
//                    v[1].position_world * (1 / v[1].position_clip.w),
//                    v[2].position_world * (1 / v[2].position_clip.w), w0, w1,
//                    w2) *
//         iw;
//     f.ndc_z = nz;
//   }
//   Vec3 getOutputColor(const LightDepthShaderIO &f) { return f.color; }
// };

// 14. MultiLight
class MultiLightPass : public IDrawPass {
  DRAW_PASS_BOILERPLATE(MultiLightPass, MultiLightBlinnPhongShader,
                        MultiLightShaderIO)
public:
  MultiLightPass(std::string n,Mat4 model_matrix, Mat4 mvp, Mat4 nm, Vec3 eye,
                 const std::vector<Light> &lights, const Texture &t, Mesh *m,
                 const Texture &normal_map,int ms = MSAA_SAMPLES_4,
                 const float (*off)[2] = MSAA_OFFSETS_4)
      : passName(n), mesh(m), msaaSamples(ms), msaaOffsets(off) {
    shader = std::make_unique<MultiLightBlinnPhongShader>(model_matrix,mvp,nm, eye, lights, t,normal_map);
  }

private:
  void initializeVertex(MultiLightShaderIO &io, const Vertex &v) {
    io.position_world = v.pos;
    io.normal_world = v.normal;
    io.uv = v.uv;
  }
  void interpolateFragment(MultiLightShaderIO &f, const MultiLightShaderIO v[3],
                           float w0, float w1, float w2, float iw, float nz,
                           const Vec3 s[3], int W, int H) {
    f.position_world =
        baryInterp(v[0].position_world * (1 / v[0].position_clip.w),
                   v[1].position_world * (1 / v[1].position_clip.w),
                   v[2].position_world * (1 / v[2].position_clip.w), w0, w1,
                   w2) *
        iw;
    f.normal_world =
        baryInterp(v[0].normal_world * (1 / v[0].position_clip.w),
                   v[1].normal_world * (1 / v[1].position_clip.w),
                   v[2].normal_world * (1 / v[2].position_clip.w), w0, w1, w2) *
        iw;
    f.tangent_world =
        baryInterp(v[0].tangent_world * (1 / v[0].position_clip.w),
                   v[1].tangent_world * (1 / v[1].position_clip.w),
                   v[2].tangent_world * (1 / v[2].position_clip.w), w0, w1,
                   w2) *
        iw;
    f.uv = baryInterp(v[0].uv * (1 / v[0].position_clip.w),
                      v[1].uv * (1 / v[1].position_clip.w),
                      v[2].uv * (1 / v[2].position_clip.w), w0, w1, w2) *
           iw;
  }
  Vec3 getOutputColor(const MultiLightShaderIO &f) { return f.color; }
};

// 15. NoMSAA (BlinnPhong with 1x MSAA)
class NoMSAAPass : public IDrawPass {
  DRAW_PASS_BOILERPLATE(NoMSAAPass, BlinnPhongShader, BlinnPhongShaderIO)
public:
  NoMSAAPass(std::string n, Mat4 model_matrix,Mat4 mvp, Mat4 nm, Vec3 eye, Vec3 l,
             const Texture &t, const Texture &tn,Mesh *m)
      : passName(n), mesh(m), msaaSamples(MSAA_SAMPLES_1),
        msaaOffsets(MSAA_OFFSETS_1) {
    shader = std::make_unique<BlinnPhongShader>(model_matrix,mvp, nm, eye, l, t,tn);
  }

private:
  void initializeVertex(BlinnPhongShaderIO &io, const Vertex &v) {
    io.position_world = v.pos;
    io.normal_world = v.normal;
    io.tangent_world = v.tangent;
    io.uv = v.uv;
  }
  void interpolateFragment(BlinnPhongShaderIO &f, const BlinnPhongShaderIO v[3],
                           float w0, float w1, float w2, float iw, float nz,
                           const Vec3 s[3], int W, int H) {
    f.position_world =
        baryInterp(v[0].position_world * (1 / v[0].position_clip.w),
                   v[1].position_world * (1 / v[1].position_clip.w),
                   v[2].position_world * (1 / v[2].position_clip.w), w0, w1,
                   w2) *
        iw;
    f.normal_world =
        baryInterp(v[0].normal_world * (1 / v[0].position_clip.w),
                   v[1].normal_world * (1 / v[1].position_clip.w),
                   v[2].normal_world * (1 / v[2].position_clip.w), w0, w1, w2) *
        iw;
    f.tangent_world =
        baryInterp(v[0].tangent_world * (1 / v[0].position_clip.w),
                   v[1].tangent_world * (1 / v[1].position_clip.w),
                   v[2].tangent_world * (1 / v[2].position_clip.w), w0, w1,
                   w2) *
        iw;
    f.uv = baryInterp(v[0].uv * (1 / v[0].position_clip.w),
                      v[1].uv * (1 / v[1].position_clip.w),
                      v[2].uv * (1 / v[2].position_clip.w), w0, w1, w2) *
           iw;
    f.ndc_z = nz;
  }
  Vec3 getOutputColor(const BlinnPhongShaderIO &f) { return f.color; }
};

// 16. Bloom
// class BloomPass : public IDrawPass {
//   DRAW_PASS_BOILERPLATE(BloomPass, BloomShader, BloomShaderIO)
// public:
//   BloomPass(std::string n, Mat4 mvp, const Texture &t, float strength, Mesh *m,
//             int ms = MSAA_SAMPLES_4, const float (*off)[2] = MSAA_OFFSETS_4)
//       : passName(n), mesh(m), msaaSamples(ms), msaaOffsets(off) {
//     shader = std::make_unique<BloomShader>(mvp, t, strength);
//   }

// private:
//   void initializeVertex(BloomShaderIO &io, const Vertex &v) {
//     io.position_world = v.pos;
//     io.uv = v.uv;
//   }
//   void interpolateFragment(BloomShaderIO &f, const BloomShaderIO v[3], float w0,
//                            float w1, float w2, float iw, float nz,
//                            const Vec3 s[3], int W, int H) {
//     f.position_world =
//         baryInterp(v[0].position_world * (1 / v[0].position_clip.w),
//                    v[1].position_world * (1 / v[1].position_clip.w),
//                    v[2].position_world * (1 / v[2].position_clip.w), w0, w1,
//                    w2) *
//         iw;
//     f.uv = baryInterp(v[0].uv * (1 / v[0].position_clip.w),
//                       v[1].uv * (1 / v[1].position_clip.w),
//                       v[2].uv * (1 / v[2].position_clip.w), w0, w1, w2) *
//            iw;
//   }
//   Vec3 getOutputColor(const BloomShaderIO &f) { return f.bloom_color; }
// };

int main() {
  const int W = 512, H = 512;
  // Load meshes and textures
  Mesh spot_m = loadModel("models/spot/spot_triangulated_good.obj");
  Texture spot_t = loadTexture("models/spot/spot_texture.png");
  Texture spot_bump = loadTexture("models/spot/hmap.jpg");
  Mesh rock_m = loadModel("models/rock/rock.obj");
  Texture rock_t = loadTexture("models/rock/rock.png");
  Mesh brick2_m = loadModel("models/bricks2/bricks2.obj");
  Texture brick2_t = loadTexture("models/bricks2/bricks2.jpg");
  Texture brick2_n = loadTexture("models/bricks2/bricks2_normal.jpg");
  Texture brick2_d = loadTexture("models/bricks2/bricks2_disp.jpg");

  Mesh cannon_m = loadModel("models/cannon/rigged_cannon.obj");
  Texture cannon_t = loadTexture("models/cannon/Cannon.png");
  Mat4 cannon_model_matrix = Mat4::identity();
  cannon_model_matrix = Mat4::scale(cannon_model_matrix, Vec3{3.0f, 3.0f, 3.0f});
  cannon_model_matrix = Mat4::translate(cannon_model_matrix, Vec3{0.3f, -0.2f, 0.0f});
  Texture cannon_normal_map = loadTexture("models/cannon/normalMap1.png");

  // Camera and matrices
  Vec3 eye = {-2, 2, -2};
  Vec3 target = {0, 0, 0};
  Vec3 up = {0, 1, 0};
  Mat4 V = Mat4::lookAt(eye, target, up);
  Mat4 P = Mat4::perspective(45.f, (float)W / H, .1f, 100.f);
  Mat4 VP = P * V;
  Mat4 N = V.inverse().transpose();

  Mat4 cannon_mvp = VP * cannon_model_matrix;
  Mat4 N_cannon = cannon_model_matrix.inverse().transpose();
  Mat4 N_V_cannon = (V*cannon_model_matrix).inverse().transpose();

  // Light for MultiLight and LightDepth
  std::vector<Light> lights = {
      {{2.0f, 2.0f, 2.0f}, {0, 0, 0}, {1, 1, 0}, 1.0f},
      {{-2.0f, 2.0f, -2.0f}, {0, 0, 0}, {0, 0, 1}, 0.8f},
      {{-2, -2, -2}, {0, 0, 0}, {0, 1, 0}, 0.8f}};
  Vec3 light_pos = {3.0f, 3.0f, 3.0f};
  Vec3 light_target = {0.0f, 0.0f, 0.0f};
  Vec3 light_up = {0.0f, 1.0f, 0.0f};
  Mat4 light_view = Mat4::lookAt(light_pos, light_target, light_up);
  Mat4 light_proj = Mat4::orthographic(-2.0f, 2.0f, -2.0f, 2.0f, 1.0f, 20.0f);
  Mat4 light_mvp = light_proj * light_view;

  DrawGraph graph;
  // 1. BlinnPhong
  graph.addPass(std::make_unique<BlinnPhongPass>(
      "output_blinn.ppm", cannon_model_matrix,cannon_mvp, N_cannon, eye, Vec3{1, 1, -1}, cannon_t,cannon_normal_map, &cannon_m));
  // 2. Normal
  graph.addPass(
      std::make_unique<NormalPass>("output_normal.ppm",cannon_model_matrix,cannon_mvp, N_V_cannon, eye, Vec3{1, 1, -1}, cannon_t,cannon_normal_map, &cannon_m));
  // 3. Texture
  graph.addPass(std::make_unique<TexturePass>("output_texture.ppm", cannon_mvp, N_cannon,
                                              cannon_t, &cannon_m));
  // 4. Bump (use rock mesh and spot bump texture)
  graph.addPass(std::make_unique<BumpPass>("output_bump.ppm",cannon_model_matrix, cannon_mvp, N_cannon, eye,
                                           Vec3{1, 1, -1}, cannon_t, spot_bump, &cannon_m));

  // // 5. Displacement (use spot mesh, brick2 disp texture)
  // graph.addPass(std::make_unique<DisplacementPass>("output_disp.ppm", VP, N_cannon,
  //                                                  eye, Vec3{1, 1, -1}, cannon_t,
  //                                                  brick2_d, 0.1f, &cannon_m));
  // // 6. Shadow
  // graph.addPass(std::make_unique<ShadowPass>("output_shadow.ppm", cannon_model_matrix, N_cannon,
  //                                            Vec3{1, 1, -1}, &cannon_m));
  // 7. AO
  graph.addPass(
      std::make_unique<AOPass>("output_ao.ppm", cannon_model_matrix,cannon_mvp, N_cannon, Vec3{0, 1, 0}, &cannon_m,cannon_normal_map));
  // // 8. AlphaBlend
  // graph.addPass(std::make_unique<AlphaBlendPass>("output_alpha.ppm", VP, N,
  //                                                spot_t, 0.7f, &spot_m));
  // 9. Parallax (use brick2 mesh and textures)
  graph.addPass(std::make_unique<ParallaxPass>(
      "output_parallax.ppm", VP, N, eye, Vec3{1, 1, -1}, brick2_t, brick2_d,
      brick2_n, 0.05f, &brick2_m));
  // 10. SSAO
  graph.addPass(std::make_unique<SSAOPass>("output_ssao.ppm",cannon_model_matrix, cannon_mvp, N_V_cannon, V, 0.1f,
                                           100.f, &cannon_m,cannon_normal_map));
  // 11. SpotLight
  graph.addPass(std::make_unique<SpotLightPass>("output_spotlight.ppm", VP, eye,
                                                Vec3{0, 5, 5}, Vec3{0, -1, -1},
                                                spot_t, 5.0f, 10.0f, &spot_m));
  // 12. Depth
  graph.addPass(std::make_unique<DepthPass>("output_depth.ppm", cannon_mvp, &cannon_m));
  // // 13. LightDepth
  // graph.addPass(std::make_unique<LightDepthPass>("output_light_depth.ppm",
  //                                                light_mvp, &cannon_m));
  // 14. MultiLight
  graph.addPass(std::make_unique<MultiLightPass>("output_multi_light.ppm", cannon_model_matrix,cannon_mvp,N_cannon,
                                                 eye, lights, cannon_t, &cannon_m,cannon_normal_map));
  // 15. NoMSAA (BlinnPhong with 1x MSAA)
  graph.addPass(std::make_unique<NoMSAAPass>("output_nomsa.ppm", cannon_model_matrix,cannon_mvp, N_cannon, eye,
                                             Vec3{1, 1, -1}, cannon_t,cannon_normal_map, &cannon_m));
  // // 16. Bloom
  // graph.addPass(std::make_unique<BloomPass>("output_bloom.ppm", VP, spot_t,
  //                                           0.8f, &spot_m));

  graph.render(W, H);
  return 0;
}