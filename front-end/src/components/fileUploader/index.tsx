import { message, Upload, Button } from 'antd';
import { UploadOutlined } from '@ant-design/icons';
import styles from './index.less';

export const FileUploader: React.FC = () => {
  const props = {
    name: 'file',
    action: 'https://www.mocky.io/v2/5cc8019d300000980a055e76',
    onChange(info: any) {
      if (info.file.status !== 'uploading') {
        console.log(info.file, info.fileList);
      }
      if (info.file.status === 'done') {
        message.success(`${info.file.name} file uploaded successfully`);
      } else if (info.file.status === 'error') {
        message.error(`${info.file.name} file upload failed.`);
      }
    },
  };
  return (
    <div className={styles.container}>
      <h2 className={styles.title}>代码文件</h2>
      <Upload {...props}>
        <Button icon={<UploadOutlined />}>点击上传</Button>
      </Upload>
    </div>
  );
};
