import { useEffect, useState, type ReactNode } from "react";
import { ArrowRightOutlined, KeyOutlined, ReloadOutlined, SafetyOutlined } from "@ant-design/icons";
import { Alert, Button, Form, Input, Spin } from "antd";
import { fetchAuthStatus, loginWithApiKey } from "../lib/api";
import "./AuthGate.css";

interface AuthGateProps {
  children: ReactNode;
}

export function AuthGate({ children }: AuthGateProps) {
  const [status, setStatus] = useState<"loading" | "login" | "ready" | "unavailable">("loading");
  const [attempt, setAttempt] = useState(0);
  const [error, setError] = useState("");
  const [pending, setPending] = useState(false);

  useEffect(() => {
    let disposed = false;
    const controller = new AbortController();
    const timeout = window.setTimeout(() => controller.abort(), 10000);
    fetchAuthStatus(controller.signal).then((result) => {
      if (!disposed) setStatus(result.authenticated ? "ready" : "login");
    }).catch(() => {
      if (!disposed) setStatus("unavailable");
    }).finally(() => window.clearTimeout(timeout));
    return () => {
      disposed = true;
      window.clearTimeout(timeout);
      controller.abort();
    };
  }, [attempt]);

  function handleRetry() {
    setError("");
    setStatus("loading");
    setAttempt((value) => value + 1);
  }

  async function handleLogin(values: { apiKey: string }) {
    setPending(true);
    setError("");
    try {
      await loginWithApiKey(values.apiKey);
      setStatus("ready");
    } catch (reason) {
      if (reason instanceof TypeError) {
        setStatus("unavailable");
      } else {
        setError(reason instanceof Error ? reason.message : "登录失败，请重新输入访问密钥。");
      }
    } finally {
      setPending(false);
    }
  }

  if (status === "ready") return children;
  return (
    <main className="auth-page">
      <section className="auth-story" aria-label="DeepSearch 深度研搜">
        <div className="auth-brand">
          <span className="brand-mark" aria-hidden="true"><i /><i /><i /></span>
          <span>DEEPSEARCH<span className="auth-brand-caption">多智能体研究工作台</span></span>
        </div>
        <div className="auth-story-body">
          <p className="auth-eyebrow">FROM QUESTION TO EVIDENCE</p>
          <h1>让每一次追问，<br />都有<span>据</span>可循。</h1>
          <p className="auth-story-description">连接信息，协同推理。<br />把复杂问题，变成清晰、可追溯的答案。</p>
          <div className="auth-constellation" aria-hidden="true">
            <div className="auth-orbit auth-orbit-outer" />
            <div className="auth-orbit auth-orbit-inner" />
            <div className="auth-core">DS<span>RESEARCH ENGINE</span></div>
            <span className="auth-node auth-node-web">01 / 公开网页</span>
            <span className="auth-node auth-node-data">02 / 业务数据</span>
            <span className="auth-node auth-node-knowledge">03 / 内部知识</span>
          </div>
        </div>
        <div className="auth-story-footer"><span>检索 · 推理 · 交付</span><span>DEEPSEARCH OS / 01</span></div>
      </section>

      <section className="auth-entry" aria-labelledby="auth-title">
        <div className="auth-entry-top"><span>你的研究，从这里开始</span><SafetyOutlined aria-hidden="true" /></div>
        <div className="auth-panel">
          <p className="auth-eyebrow">RESEARCH WORKSPACE</p>
          <div className={`auth-status auth-status-${status}`} role="status">
            <span className="auth-status-dot" />
            {status === "loading" ? "正在连接服务" : status === "unavailable" ? "服务暂时无法连接" : "工作台已启用访问验证"}
          </div>
          <h2 id="auth-title">{status === "unavailable" ? "连接暂时中断" : status === "loading" ? "正在打开工作台" : "欢迎进入深度研搜"}</h2>
          {status === "loading" && <div className="auth-loading" role="status"><Spin /><p>正在检查工作台状态，请稍候。</p></div>}
          {status === "unavailable" && <>
            <p className="auth-description">暂时无法确认工作台状态。请检查后端服务和网络连接，然后重试。</p>
            <div className="auth-connection-note">点击重新连接后会再次确认是否需要登录，你现在无需输入访问密钥。</div>
            <Button className="auth-submit" size="large" type="primary" block icon={<ReloadOutlined />} onClick={handleRetry}>重新连接</Button>
          </>}
          {status === "login" && <>
            <p className="auth-description">输入你的访问密钥，继续探索问题背后的答案。</p>
            {error && <Alert className="auth-error" type="error" message={error} showIcon role="alert" />}
            <Form onFinish={handleLogin} layout="vertical" requiredMark={false} disabled={pending}>
              <Form.Item label="访问密钥" name="apiKey" rules={[{ required: true, whitespace: true, message: "请输入管理员分配的访问密钥" }]}>
                <Input.Password size="large" prefix={<KeyOutlined aria-hidden="true" />} placeholder="输入你的工作台访问密钥" autoComplete="current-password" />
              </Form.Item>
              <Button className="auth-submit" size="large" type="primary" htmlType="submit" block loading={pending} icon={<ArrowRightOutlined />} iconPosition="end">进入工作台</Button>
            </Form>
            <details className="auth-help">
              <summary>访问密钥在哪里？</summary>
              <p>团队成员请向部署管理员获取。自行部署时，在项目根目录 <code>.env</code> 的 <code>AUTH_API_KEY</code> 或 <code>AUTH_USERS</code> 中配置。</p>
              <p>这是工作台的登录凭证，请勿填写模型服务的 API Key。关闭认证的本地环境会直接进入工作台。</p>
            </details>
          </>}
        </div>
        <footer className="auth-entry-footer">一个问题，调动整支研究团队。<span>POWERED BY DEEPSEARCH AGENTS</span></footer>
      </section>
    </main>
  );
}
