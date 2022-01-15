import { Col, Result as AntdResult, Row, Statistic } from 'antd';
import { Result as ResultProps } from '@/models/code';

export const Result: React.FC<{ result: ResultProps }> = ({ result }) => {
  const { data } = result;
  return (
    <AntdResult
      status={result.error ? 'error' : result.flag ? 'warning' : 'success'}
      title={result.error ? '检测失败：' + result.message : result.message}
      extra={[
        <Row gutter={16} justify="center" align="middle">
          <Col span={9}>
            <Statistic
              title={'Type-1 clone chance'}
              value={data['type-1']}
              precision={4}
            />
          </Col>
          <Col span={9}>
            <Statistic
              title={'Type-2 clone chance'}
              value={data['type-2']}
              precision={4}
            />
          </Col>
        </Row>,
      ]}
    />
  );
};
