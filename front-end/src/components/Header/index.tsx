import { Layout } from 'antd';
import styles from './index.less';

export const Header: React.FC = () => {
  const pageTitle = "代码克隆检测助手";
  return <Layout.Header className={styles.header}>
    <h1 className={styles.pageTitle}>{pageTitle}</h1>
  </Layout.Header>
}
