import "antd/dist/reset.css";
import { App as AntApp, ConfigProvider, theme } from "antd";
import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import "./styles.css";

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <ConfigProvider
      theme={{
        algorithm: theme.defaultAlgorithm,
        token: {
          colorPrimary: "#0284c7",
          colorSuccess: "#059669",
          colorWarning: "#d97706",
          colorError: "#e11d48",
          colorInfo: "#6366f1",
          colorText: "#14263c",
          colorTextSecondary: "#5c6e82",
          colorBgBase: "#f4f8fc",
          colorBgContainer: "rgba(255, 255, 255, 0.9)",
          colorBorder: "rgba(2, 132, 199, 0.18)",
          borderRadius: 8,
          fontFamily:
            "'IBM Plex Sans', 'PingFang SC', 'Microsoft YaHei', system-ui, sans-serif",
          fontFamilyCode:
            "'JetBrains Mono', 'SFMono-Regular', Consolas, 'Liberation Mono', monospace"
        },
        components: {
          Button: {
            controlHeightLG: 46,
            primaryShadow: "0 0 24px rgba(2, 132, 199, 0.22)"
          },
          Input: {
            activeBorderColor: "#0284c7",
            hoverBorderColor: "#059669"
          }
        }
      }}
    >
      <AntApp>
        <App />
      </AntApp>
    </ConfigProvider>
  </React.StrictMode>
);
