import { FileUploader } from '@/components/fileUploader';
import { Layout, Button, Progress } from 'antd';
import { useState } from 'react';
import styles from './index.less';

export default function IndexPage() {
  const pageTitle = '代码克隆检测助手';
  const [percent, setPercent] = useState(0);

  const startTest = () => {
    let progress = 0;
    const p = setInterval(() => {
      if (percent === 100) {
        clearInterval(p);
      }
      setPercent(progress++);
    }, 60);
  };

  return (
    <Layout className={styles.wrapper}>
      <Layout.Header className={styles.header}>
        <h1 className={styles.pageTitle}>{pageTitle}</h1>
      </Layout.Header>
      <Layout.Content className={styles.content}>
        <div className={styles.uploaders}>
          <FileUploader />
          <FileUploader />
        </div>
        <Button onClick={startTest} className={styles.startButton}>
          开始检测
        </Button>
        <Progress
          className={styles.progress}
          type="circle"
          percent={percent}
        ></Progress>
      </Layout.Content>
      <Layout.Footer></Layout.Footer>
    </Layout>
  );
}
