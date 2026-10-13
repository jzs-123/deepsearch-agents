import "antd/dist/reset.css";
import { App as AntApp, ConfigProvider, theme } from "antd";
import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import "./styles.css";
import "./workbench.css";

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <ConfigProvider
      theme={{
        algorithm: theme.defaultAlgorithm,
        token: {
          colorPrimary: "#596bff",
          colorSuccess: "#34bb8f",
          colorWarning: "#e37a51",
          colorError: "#e11d48",
          colorInfo: "#6366f1",
          colorText: "#172033",
          colorTextSecondary: "#687386",
          colorBgBase: "#f5f4ef",
          colorBgContainer: "rgba(255, 255, 255, 0.9)",
          colorBorder: "rgba(32, 42, 60, 0.12)",
          borderRadius: 14,
          fontFamily:
            "'IBM Plex Sans', 'PingFang SC', 'Microsoft YaHei', system-ui, sans-serif",
          fontFamilyCode:
            "'JetBrains Mono', 'SFMono-Regular', Consolas, 'Liberation Mono', monospace"
        },
        components: {
          Button: {
            controlHeightLG: 46,
            primaryShadow: "0 12px 30px rgba(89, 107, 255, 0.22)"
          },
          Input: {
            activeBorderColor: "#596bff",
            hoverBorderColor: "#34bb8f"
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
