# Genesis Web UI - 文生图工具

基于 Ant Design Pro 的 Genesis AI 图像生成前端界面。

## 功能特性

- 🎨 **文生图**: 通过文字描述生成图像
- 📊 **实时进度**: 实时显示生成进度
- 🎯 **参数控制**: 完整的生成参数配置
- 📜 **历史记录**: 查看所有生成历史
- 🚀 **现代化UI**: 基于 Ant Design Pro

## 技术栈

- React 18
- TypeScript
- Ant Design Pro
- Vite
- Zustand (状态管理)
- Axios (HTTP 客户端)

## 快速开始

### 1. 安装依赖

```bash
npm install
# 或
yarn install
# 或
pnpm install
```

### 2. 启动后端服务

确保 Genesis 后端服务正在运行：

```bash
cd ../Genesis-main
python api/advanced_server.py
```

后端服务默认运行在 `http://localhost:5000`

### 3. 启动前端开发服务器

```bash
npm run dev
```

前端服务将运行在 `http://localhost:8000`

### 4. 构建生产版本

```bash
npm run build
```

## 项目结构

```
genesis-web-ui/
├── src/
│   ├── layouts/          # 布局组件
│   ├── pages/            # 页面组件
│   │   ├── TextToImage/  # 文生图页面
│   │   ├── TaskHistory/  # 历史记录页面
│   │   └── Settings/     # 设置页面
│   ├── services/         # API 服务
│   ├── store/            # 状态管理
│   ├── App.tsx           # 应用入口
│   └── main.tsx          # 主入口
├── package.json
├── vite.config.ts
└── tsconfig.json
```

## 使用说明

1. **文生图**
   - 输入正向提示词描述想要生成的图像
   - 可选输入负向提示词避免不想要的内容
   - 调整参数（尺寸、步数、CFG等）
   - 点击"开始生成"按钮
   - 等待生成完成

2. **历史记录**
   - 查看所有生成的图像
   - 点击图像可查看大图

3. **设置**
   - 查看系统信息
   - 查看设备状态

## API 接口

前端通过 `/api` 代理访问后端服务：

- `POST /api/task/submit` - 提交生成任务
- `GET /api/task/:id` - 查询任务状态
- `GET /api/models` - 获取可用模型
- `GET /api/device` - 获取设备信息

## 开发说明

- 使用 Vite 作为构建工具，支持快速热更新
- 使用 TypeScript 提供类型安全
- 使用 Zustand 进行轻量级状态管理
- 使用 Ant Design Pro 组件库

## License

MIT
