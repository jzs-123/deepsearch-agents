import { InboxOutlined } from "@ant-design/icons";
import { Alert, AutoComplete, Button, Drawer, Form, Input, Progress, Upload } from "antd";
import type { UploadFile } from "antd";
import { useEffect, useState } from "react";
import type { ChangeEvent } from "react";
import { ingestKnowledgeBase, listKnowledgeBases } from "../lib/api";
import type { IngestReport, KnowledgeBaseSummary } from "../types";

interface KnowledgeBaseManagerProps {
  open: boolean;
  onClose: () => void;
}

export function KnowledgeBaseManager({ open, onClose }: KnowledgeBaseManagerProps) {
  const [bases, setBases] = useState<KnowledgeBaseSummary[]>([]);
  const [kbName, setKbName] = useState("");
  const [fileList, setFileList] = useState<UploadFile[]>([]);
  const [documentKey, setDocumentKey] = useState("");
  const [revisionId, setRevisionId] = useState("");
  const [effectiveFrom, setEffectiveFrom] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [report, setReport] = useState<IngestReport | null>(null);

  async function refresh() {
    try {
      const response = await listKnowledgeBases();
      setBases(response.knowledge_bases);
      setKbName((current) => current || response.default_kb);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "知识库列表加载失败");
    }
  }

  useEffect(() => {
    if (open) {
      setError("");
      void refresh();
    }
  }, [open]);

  async function handleIngest() {
    const files = fileList
      .map((item) => item.originFileObj as unknown as File | undefined)
      .filter((item): item is File => Boolean(item));
    if (!kbName.trim() || files.length === 0) {
      setError("请填写知识库名称并选择文件");
      return;
    }

    setLoading(true);
    setError("");
    setReport(null);
    try {
      const response = await ingestKnowledgeBase({
        kbName: kbName.trim(),
        files,
        documentKey: files.length === 1 ? documentKey.trim() : "",
        revisionId: files.length === 1 ? revisionId.trim() : "",
        effectiveFrom: files.length === 1 ? effectiveFrom.trim() : "",
      });
      setReport(response.report);
      setFileList([]);
      await refresh();
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "入库失败");
    } finally {
      setLoading(false);
    }
  }

  const current = bases.find((item) => item.name === kbName.trim());

  return (
    <Drawer
      className="kb-drawer"
      destroyOnClose={false}
      onClose={onClose}
      open={open}
      placement="left"
      title="知识库管理"
      width={460}
    >
      <p className="kb-intro">
        上传后立即切块并建立索引。同一文件名会作为同一逻辑文档，新内容自动替换旧版本。
      </p>

      <Form layout="vertical">
        <Form.Item label="知识库名称" required>
          <AutoComplete
            onChange={setKbName}
            options={bases.map((item) => ({ value: item.name }))}
            placeholder="选择已有知识库，或输入新名称"
            value={kbName}
          />
        </Form.Item>

        {current ? (
          <div className="kb-current-card">
            <strong>{current.chunk_count.toLocaleString()} 个片段</strong>
            <span>{current.embedding || "尚未初始化向量模型"}</span>
            <small>{current.recommendation}</small>
          </div>
        ) : null}

        <Form.Item label="文档">
          <Upload.Dragger
            beforeUpload={() => false}
            capture={undefined}
            fileList={fileList}
            hasControlInside
            multiple
            onChange={({ fileList: next }) => setFileList(next)}
            pastable={false}
          >
            <InboxOutlined className="kb-upload-icon" />
            <p>点击或拖入 PDF、Word、Markdown、文本、CSV 或图片</p>
            <span>单文件上限 50MB，可一次上传多个文件</span>
          </Upload.Dragger>
        </Form.Item>

        {fileList.length === 1 ? (
          <div className="kb-version-fields">
            <Form.Item label="逻辑文档标识（可选）">
              <Input
                onChange={(event: ChangeEvent<HTMLInputElement>) => setDocumentKey(event.target.value)}
                placeholder="例如 school/放学时间"
                value={documentKey}
              />
            </Form.Item>
            <Form.Item label="版本号（可选）">
              <Input
                onChange={(event: ChangeEvent<HTMLInputElement>) => setRevisionId(event.target.value)}
                placeholder="例如 2026-10-13"
                value={revisionId}
              />
            </Form.Item>
            <Form.Item label="生效日期（可选）">
              <Input
                onChange={(event: ChangeEvent<HTMLInputElement>) => setEffectiveFrom(event.target.value)}
                placeholder="例如 2026-10-13"
                value={effectiveFrom}
              />
            </Form.Item>
          </div>
        ) : null}
      </Form>

      {loading ? <Progress percent={99} showInfo={false} status="active" /> : null}
      {error ? <Alert message={error} showIcon type="error" /> : null}
      {report ? (
        <Alert
          message={`入库完成：新增 ${report.chunks_added} 个片段，替换 ${report.chunks_replaced} 个旧片段`}
          description={`当前知识库共 ${report.total_chunks} 个片段，用时 ${report.elapsed} 秒`}
          showIcon
          type="success"
        />
      ) : null}

      <Button
        block
        className="kb-submit"
        disabled={!kbName.trim() || fileList.length === 0}
        loading={loading}
        onClick={() => void handleIngest()}
        size="large"
        type="primary"
      >
        上传并建立索引
      </Button>
    </Drawer>
  );
}
