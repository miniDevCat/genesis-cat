import React, { useState } from 'react'
import { useNavigate, useLocation } from 'react-router-dom'
import { ProLayout } from '@ant-design/pro-components'
import {
  PictureOutlined,
  HistoryOutlined,
  SettingOutlined,
  RocketOutlined,
  BgColorsOutlined,
  VideoCameraOutlined,
} from '@ant-design/icons'

interface BasicLayoutProps {
  children: React.ReactNode
}

const BasicLayout: React.FC<BasicLayoutProps> = ({ children }) => {
  const navigate = useNavigate()
  const location = useLocation()
  const [pathname, setPathname] = useState(location.pathname)

  const menuItems = [
    {
      path: '/text-to-image',
      name: '文生图',
      icon: <PictureOutlined />,
    },
    {
      path: '/text-to-video',
      name: '文生视频',
      icon: <VideoCameraOutlined />,
    },
    {
      path: '/history',
      name: '历史记录',
      icon: <HistoryOutlined />,
    },
    {
      path: '/themes',
      name: '主题皮肤',
      icon: <BgColorsOutlined />,
    },
    {
      path: '/settings',
      name: '设置',
      icon: <SettingOutlined />,
    },
  ]

  return (
    <ProLayout
      title="Genesis AI"
      logo={<RocketOutlined style={{ fontSize: 32 }} />}
      layout="mix"
      splitMenus={false}
      navTheme="light"
      contentWidth="Fluid"
      fixedHeader
      fixSiderbar
      location={{
        pathname,
      }}
      route={{
        path: '/',
        routes: menuItems,
      }}
      menuItemRender={(item, dom) => (
        <div
          onClick={() => {
            setPathname(item.path || '/')
            navigate(item.path || '/')
          }}
        >
          {dom}
        </div>
      )}
      headerContentRender={() => (
        <div style={{ fontSize: 16, fontWeight: 500, color: '#666' }}>
          AI 图像生成工具
        </div>
      )}
    >
      <div style={{ minHeight: 'calc(100vh - 48px)' }}>{children}</div>
    </ProLayout>
  )
}

export default BasicLayout
