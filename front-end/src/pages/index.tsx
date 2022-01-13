import { Button, Layout } from 'antd';
import { useState } from 'react';
import { CodeUploader, Footer, Header } from '@/components';
import { detectClone } from '@/models/code';
import styles from './index.less';

export default function IndexPage() {
  const [codeX, setCodeX] = useState('');
  const [codeY, setCodeY] = useState('');

  return (
    <Layout className={styles.wrapper}>
      <Header />
      <Layout.Content className={styles.content}>
        <div className={styles.uploaders}>
          <CodeUploader code={codeX} setCode={setCodeX} />
          <CodeUploader code={codeY} setCode={setCodeY} />
        </div>
        <Button
          className={styles.startButton}
          onClick={() => detectClone([codeX, codeY])}
        >
          开始检测
        </Button>
      </Layout.Content>
      <Footer />
    </Layout>
  );
}
