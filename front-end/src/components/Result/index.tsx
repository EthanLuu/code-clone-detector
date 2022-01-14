import { Col, Row, Statistic } from 'antd';
import { Result as ResultProps } from '@/models/code';
import { CheckCircleTwoTone, CloseCircleTwoTone, WarningTwoTone } from '@ant-design/icons';
import styles from './index.less';

export const Result: React.FC<{ result: ResultProps }> = ({ result }) => {
  if (result.error) {
    return (
      <Row gutter={16}>
        <Col span={2} className={styles.iconContainer}>
          <CloseCircleTwoTone className={styles.icon} twoToneColor={"#ff4d4f"}/>
        </Col>
        <Col>
          <Statistic
            title={'检测失败'}
            value={result.message}
          />
        </Col>
      </Row>
    );
  }

  return (
    <Row gutter={16}>
      <Col span={2} className={styles.iconContainer}>
        {result.flag ? (
          <WarningTwoTone twoToneColor="#eb2f96" className={styles.icon} />
        ) : (
          <CheckCircleTwoTone twoToneColor="#52c41a" className={styles.icon} />
        )}
      </Col>
      <Col>
        <Statistic
          title={'检测结果'}
          value={result.message}
        />
      </Col>
    </Row>
  );
};
