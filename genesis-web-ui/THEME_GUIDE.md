# Genesis Web UI - 主题皮肤系统

## 🎨 功能特性

### 6 种精美主题

1. **InfiniteTalk** - 紫色渐变，优雅现代
   - 主色：紫色 (#667eea → #764ba2)
   - 风格：优雅、现代、专业

2. **WanAnimate** - 清新蓝绿，活力十足
   - 主色：蓝绿 (#06b6d4 → #3b82f6)
   - 风格：清新、活力、科技

3. **Standard I2V** - 经典灰色，专业稳重
   - 主色：灰色 (#64748b → #475569)
   - 风格：经典、稳重、商务

4. **Cyberpunk** - 赛博朋克，炫酷科技
   - 主色：粉紫 (#ff0080 → #7928ca)
   - 风格：炫酷、科技、未来
   - 特效：发光动画

5. **Sunset** - 日落橙红，温暖热情
   - 主色：橙红 (#f59e0b → #ef4444)
   - 风格：温暖、热情、活力

6. **Forest** - 森林绿色，自然清新
   - 主色：绿色 (#10b981 → #059669)
   - 风格：自然、清新、舒适

## 📁 文件结构

```
genesis-web-ui/
├── src/
│   ├── styles/
│   │   └── themes.css              # 主题样式定义
│   ├── components/
│   │   └── ThemeSwitcher.tsx       # 主题切换组件
│   └── pages/
│       └── ThemeSettings/
│           └── index.tsx           # 主题设置页面
```

## 🚀 使用方法

### 1. 在应用中切换主题

访问 "主题皮肤" 页面，点击任意主题卡片即可切换。

### 2. 主题自动保存

选择的主题会自动保存到浏览器 localStorage，下次访问时自动应用。

### 3. 应用主题样式

所有组件都会自动应用当前主题的样式：

```tsx
// 使用主题按钮
<button className="theme-button">生成图像</button>

// 使用主题标签页
<div className="theme-tabs">
  <button className="theme-tab active">文生图</button>
  <button className="theme-tab">历史记录</button>
</div>

// 使用主题卡片
<div className="theme-card">
  内容
</div>

// 使用主题进度条
<div className="theme-progress">
  <div className="theme-progress-bar" style={{ width: '60%' }} />
</div>
```

## 🎯 自定义主题

### 添加新主题

1. 在 `themes.css` 中添加新主题类：

```css
.theme-my-custom {
  --primary-color: #your-color;
  --primary-gradient: linear-gradient(135deg, #color1 0%, #color2 100%);
  --primary-hover: #hover-color;
  --tab-active-bg: linear-gradient(135deg, #color1 0%, #color2 100%);
  --tab-active-text: #ffffff;
  --tab-inactive-text: #64748b;
  --border-radius: 12px;
  --shadow-color: rgba(r, g, b, 0.3);
}
```

2. 在 `ThemeSwitcher.tsx` 中添加主题配置：

```tsx
const themes: Theme[] = [
  // ... 现有主题
  {
    id: 'my-custom',
    name: 'My Custom Theme',
    description: '我的自定义主题',
    preview: 'linear-gradient(135deg, #color1 0%, #color2 100%)'
  }
]
```

### 修改现有主题

直接编辑 `themes.css` 中对应主题的 CSS 变量即可。

## 🎨 主题变量说明

| 变量名 | 说明 | 示例 |
|--------|------|------|
| `--primary-color` | 主色调 | `#6366f1` |
| `--primary-gradient` | 主渐变色 | `linear-gradient(...)` |
| `--primary-hover` | 悬停颜色 | `#5558e3` |
| `--tab-active-bg` | 激活标签背景 | `linear-gradient(...)` |
| `--tab-active-text` | 激活标签文字 | `#ffffff` |
| `--tab-inactive-text` | 未激活标签文字 | `#64748b` |
| `--border-radius` | 圆角大小 | `12px` |
| `--shadow-color` | 阴影颜色 | `rgba(102, 126, 234, 0.3)` |

## 💡 最佳实践

### 1. 保持一致性

所有 UI 组件都应使用主题变量，而不是硬编码颜色：

```tsx
// ✅ 好的做法
<button className="theme-button">按钮</button>

// ❌ 避免
<button style={{ background: '#6366f1' }}>按钮</button>
```

### 2. 响应式设计

主题系统支持暗色模式，会自动适配用户系统偏好。

### 3. 性能优化

- 主题切换使用 CSS 类，性能优秀
- 无需重新渲染整个应用
- 主题保存在 localStorage，加载快速

## 🌟 特殊效果

### Cyberpunk 主题

赛博朋克主题包含特殊的发光动画效果：

```css
.theme-cyberpunk .theme-tab.active {
  box-shadow: var(--glow-effect);
  animation: pulse 2s infinite;
}
```

### 进度条动画

所有主题的进度条都包含流光效果：

```css
.theme-progress-bar::after {
  animation: shimmer 2s infinite;
}
```

## 📱 移动端支持

主题系统完全支持移动端，所有主题在小屏幕上都有良好的显示效果。

## 🔧 技术细节

### 实现原理

1. 使用 CSS 变量定义主题
2. 通过切换 body 的 class 来应用主题
3. localStorage 保存用户选择
4. React Context 管理主题状态（可选）

### 浏览器兼容性

- ✅ Chrome 49+
- ✅ Firefox 31+
- ✅ Safari 9.1+
- ✅ Edge 15+

## 🎉 效果展示

访问应用后：
1. 点击左侧菜单 "主题皮肤"
2. 选择任意主题
3. 立即看到整个应用的样式变化
4. 所有页面都会应用新主题

## 📝 更新日志

### v1.0.0 (2024-11-13)
- ✨ 初始版本
- 🎨 6 种精美主题
- 💾 自动保存用户选择
- 🌙 暗色模式支持
- ✨ 特殊动画效果

---

**享受你的个性化 Genesis Web UI！** 🚀
