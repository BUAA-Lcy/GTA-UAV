import numpy as np

def build_camera_intrinsics(H, W, fov_deg):
    """
    根据图像尺寸和视场角(FOV)构建相机内参。
    
    参数:
        H (int): 图像高度
        W (int): 图像宽度
        fov_deg (float): 水平视场角，单位为度(Degree)
        
    返回:
        fx (float): x方向焦距
        fy (float): y方向焦距
        cx (float): x方向主点坐标
        cy (float): y方向主点坐标
    """
    fov_rad = np.radians(fov_deg)
    # 假设 fx = fy，根据水平视场角计算 fx
    fx = (W / 2.0) / np.tan(fov_rad / 2.0)
    fy = fx
    
    # 主点默认在图像中心
    cx = W / 2.0
    cy = H / 2.0
    
    return fx, fy, cx, cy

def build_rotation_matrix(pitch_deg, roll_deg):
    """
    构建相机到世界坐标系的旋转矩阵。
    
    坐标系约定:
    - 世界坐标系(World): X指东(East)，Y指北(North)，Z指上(Up)
    - 基础无人机相机坐标系(Base Camera): 初始假设无人机朝向正北(Heading=0, Y_w), 
      此时相机前向(Z_c) = Y_w, 相机右向(X_c) = X_w, 相机下向(Y_c) = -Z_w
    - 姿态角约定:
      - Pitch: 绕无人机横轴(X_w)旋转。Pitch=-90度表示正下视(Nadir)。
      - Roll: 绕无人机纵轴(Y_w)旋转。Roll>0表示右翼下倾。
    
    参数:
        pitch_deg (float): 俯仰角，单位为度
        roll_deg (float): 横滚角，单位为度
        
    返回:
        R (np.ndarray): 3x3 旋转矩阵，用于将相机射线转换到世界坐标系
    """
    p = np.radians(pitch_deg)
    r = np.radians(roll_deg)
    
    # 基础旋转 R_0: 相机Z指向北(+Y_w)，X指向东(+X_w)，Y指向下(-Z_w)
    R_0 = np.array([
        [1,  0,  0],
        [0,  0,  1],
        [0, -1,  0]
    ])
    
    # Pitch旋转: 绕世界X轴旋转
    R_pitch = np.array([
        [1,         0,          0],
        [0, np.cos(p), -np.sin(p)],
        [0, np.sin(p),  np.cos(p)]
    ])
    
    # Roll旋转: 绕世界Y轴旋转
    R_roll = np.array([
        [ np.cos(r), 0, np.sin(r)],
        [         0, 1,         0],
        [-np.sin(r), 0, np.cos(r)]
    ])
    
    # 组合旋转: 按照先 Pitch 后 Roll 的顺序，或者说在世界坐标系下先绕 X 轴 pitch，再绕 Y 轴 roll
    # 这样当 pitch=-90 (下视) 时，roll 会使相机光轴向左右倾斜，从而产生左右不对称的畸变
    # R @ ray_c = R_roll @ R_pitch @ R_0 @ ray_c
    R = R_roll @ R_pitch @ R_0
    
    return R

def pose_to_ground_projection(H, W, height, pitch_deg, roll_deg, fov_deg):
    """
    将图像像素投影到地面(Z=0)的网格坐标。
    
    参数:
        H, W (int): 图像尺寸
        height (float): 无人机高度 (Z轴坐标)
        pitch_deg, roll_deg (float): 姿态角
        fov_deg (float): 视场角
        
    返回:
        grid_xy (np.ndarray): [H, W, 2], 每个像素对应的地面坐标(Xg, Yg)
        valid_mask (np.ndarray): [H, W], 布尔掩码，表示射线是否与地面相交
    """
    fx, fy, cx, cy = build_camera_intrinsics(H, W, fov_deg)
    R = build_rotation_matrix(pitch_deg, roll_deg)
    
    # 1. 构造像素网格
    u = np.arange(W)
    v = np.arange(H)
    uu, vv = np.meshgrid(u, v)  # uu: [H, W], vv: [H, W]
    
    # 2. 像素到相机坐标系射线
    x_c = (uu - cx) / fx
    y_c = (vv - cy) / fy
    z_c = np.ones_like(x_c)
    rays_c = np.stack([x_c, y_c, z_c], axis=-1)  # [H, W, 3]
    
    # 3. 相机坐标系射线转换到世界坐标系
    rays_w = rays_c @ R.T  # 等价于 R @ rays_c.T 的逐像素计算, [H, W, 3]
    
    # 4. 求交点
    # 相机中心 C = [0, 0, height]
    # 射线方程 P(t) = C + t * rays_w
    # 地面方程 Z = 0  => height + t * rays_w[z] = 0
    ray_w_z = rays_w[..., 2]
    
    # valid_mask: 射线必须朝下相交，即 ray_w_z < 0
    valid_mask = ray_w_z < -1e-6
    
    # 安全计算t，避免除以0
    safe_ray_w_z = np.where(valid_mask, ray_w_z, -1e-6)
    t = -height / safe_ray_w_z
    
    # 计算地面交点 Xg, Yg
    Xg = t * rays_w[..., 0]
    Yg = t * rays_w[..., 1]
    
    grid_xy = np.stack([Xg, Yg], axis=-1)  # [H, W, 2]
    
    # 无效区域坐标置零
    grid_xy[~valid_mask] = 0.0
    
    return grid_xy, valid_mask

def compute_distortion_map(grid_xy, valid_mask):
    """
    计算地面投影网格的畸变强度图 (局部面积缩放)。
    
    原理: 
    通过有限差分求 Xg, Yg 对 u, v 的雅可比矩阵(Jacobian)，
    其行列式的绝对值即为局部面积的放大倍数。
    
    参数:
        grid_xy (np.ndarray): [H, W, 2] 的地面坐标网格
        valid_mask (np.ndarray): [H, W] 的有效像素掩码
        
    返回:
        distortion (np.ndarray): [H, W] 的畸变强度图
    """
    Xg = grid_xy[..., 0]
    Yg = grid_xy[..., 1]
    
    # 对 u (宽方向) 求导: dX/du, dY/du
    dX_du = np.gradient(Xg, axis=1)
    dY_du = np.gradient(Yg, axis=1)
    
    # 对 v (高方向) 求导: dX/dv, dY/dv
    dX_dv = np.gradient(Xg, axis=0)
    dY_dv = np.gradient(Yg, axis=0)
    
    # 雅可比行列式 J = (dX/du * dY/dv) - (dX/dv * dY/du)
    det_J = dX_du * dY_dv - dX_dv * dY_du
    distortion = np.abs(det_J)
    
    # 边界差分可能使用到无效区域(valid_mask=False)的 0.0 值导致数值异常
    # 为了稳定，我们将自身和邻居存在无效的像素都 mask 掉
    # 这里通过简单的位移来判断 4-邻居是否都 valid
    valid_y = valid_mask & np.roll(valid_mask, 1, axis=0) & np.roll(valid_mask, -1, axis=0)
    valid_x = valid_mask & np.roll(valid_mask, 1, axis=1) & np.roll(valid_mask, -1, axis=1)
    strict_valid_mask = valid_y & valid_x
    
    # 修复 roll 带来的边界回绕问题
    strict_valid_mask[0, :] = False
    strict_valid_mask[-1, :] = False
    strict_valid_mask[:, 0] = False
    strict_valid_mask[:, -1] = False
    
    distortion[~strict_valid_mask] = 0.0
    
    return distortion

def pose_to_grid(H, W, height, pitch_deg, roll_deg, fov_deg):
    """
    整体计算投影网格坐标和畸变图。
    
    参数:
        H, W (int): 图像尺寸
        height (float): 相机高度
        pitch_deg, roll_deg (float): 姿态角
        fov_deg (float): 视场角
        
    返回:
        grid_xy (np.ndarray): [H, W, 2], 地面投影坐标
        distortion (np.ndarray): [H, W], 畸变强度
        valid_mask (np.ndarray): [H, W], 是否相交掩码
    """
    grid_xy, valid_mask = pose_to_ground_projection(H, W, height, pitch_deg, roll_deg, fov_deg)
    distortion = compute_distortion_map(grid_xy, valid_mask)
    
    return grid_xy, distortion, valid_mask
