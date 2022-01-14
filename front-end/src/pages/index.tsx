import { Button, Layout, message, Modal } from 'antd';
import { useState } from 'react';
import { CodeUploader, Footer, Header, Result } from '@/components';
import { detectClone } from '@/models/code';
import styles from './index.less';

export default function IndexPage() {
  const [codeX, setCodeX] = useState('');
  const [codeY, setCodeY] = useState('');

  const handleSubmit = async () => {
    const stopLoading = message.loading('正在进行检测');
    const result = await detectClone([codeX, codeY]);
    stopLoading();
    Modal.info({
      width: 640,
      icon: null,
      closable: true,
      okText: '确认',
      content: <Result result={result} />,
    });
  };

  return (
    <Layout className={styles.wrapper}>
      <Header />
      <Layout.Content className={styles.content}>
        <div className={styles.uploaders}>
          <CodeUploader code={codeX} setCode={setCodeX} />
          <CodeUploader code={codeY} setCode={setCodeY} />
        </div>
        <Button className={styles.startButton} onClick={handleSubmit}>
          开始检测
        </Button>
      </Layout.Content>
      <Footer />
    </Layout>
  );
}
