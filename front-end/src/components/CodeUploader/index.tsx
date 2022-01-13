import { Input, Upload } from 'antd';
import React from 'react';
import { InboxOutlined } from '@ant-design/icons';
import styles from './index.less';

export const CodeUploader: React.FC<{
  code: string;
  setCode: React.Dispatch<React.SetStateAction<string>>;
}> = ({ code, setCode }) => {
  const handleInput = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setCode(e.currentTarget.value);
  };

  const readFile = (file: any) => {
    const reader = new FileReader();
    reader.onload = function () {
      setCode(reader.result as string);
    };
    reader.readAsText(file);
    return false;
  };
  return (
    <div className={styles.container}>
      <h2 className={styles.title}>代码文件</h2>
      <div className={styles.dragger}>
        <Upload.Dragger
          name="file"
          beforeUpload={readFile}
          maxCount={1}
          onRemove={() => setCode('')}
        >
          <p className="ant-upload-drag-icon">
            <InboxOutlined />
          </p>
          <span className="ant-upload-text">点击上传或拖拽至此处</span>
        </Upload.Dragger>
      </div>

      <Input.TextArea
        placeholder="请上传代码文件或将文本复制此处"
        value={code}
        onChange={handleInput}
        allowClear
        rows={12}
      ></Input.TextArea>
    </div>
  );
};
