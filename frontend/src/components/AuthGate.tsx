import { useEffect, useState, type ReactNode } from "react";
import { Alert, Button, Card, Form, Input, Spin } from "antd";
import { fetchAuthStatus, loginWithApiKey } from "../lib/api";

interface AuthGateProps {
  children: ReactNode;
}

export function AuthGate({ children }: AuthGateProps) {
  const [status, setStatus] = useState<"loading" | "login" | "ready">("loading");
  const [error, setError] = useState("");
  const [pending, setPending] = useState(false);

  useEffect(() => {
    let disposed = false;
    fetchAuthStatus().then((result) => {
      if (!disposed) setStatus(result.authenticated ? "ready" : "login");
    }).catch((reason: unknown) => {
      if (!disposed) {
        setStatus("login");
        setError(reason instanceof Error ? reason.message : "无法连接服务");
      }
    });
    return () => { disposed = true; };
  }, []);

  async function handleLogin(values: { apiKey: string }) {
    setPending(true);
    setError("");
    try {
      await loginWithApiKey(values.apiKey);
      setStatus("ready");
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "登录失败");
    } finally {
      setPending(false);
    }
  }

  if (status === "ready") return children;
  if (status === "loading") return <Spin fullscreen tip="正在连接研究工作台" />;

  return (
    <Card title="登录研究工作台">
      <p>请输入管理员为你分配的访问密钥。</p>
      {error && <Alert type="error" message={error} showIcon />}
      <Form onFinish={handleLogin} layout="vertical">
        <Form.Item label="访问密钥" name="apiKey" rules={[{ required: true, message: "请输入访问密钥" }]}>
          <Input.Password autoComplete="current-password" />
        </Form.Item>
        <Button type="primary" htmlType="submit" loading={pending}>进入工作台</Button>
      </Form>
    </Card>
  );
}
