export interface Result {
  flag: boolean;
  message: string;
  error?: boolean;
  data: {
    'type-1'?: number;
    'type-2'?: number;
  };
}

const api =
  process.env.NODE_ENV === 'development' ? '' : 'https://cloud.ethanloo.cn';
export const detectClone = async (samples: string[]) => {
  if (samples.length < 2) {
    return;
  }
  const result: Result = {
    flag: false,
    message: '',
    data: {},
  };
  return await fetch(`${api}/detect`, {
    method: 'POST',
    body: JSON.stringify({
      codeX: samples[0],
      codeY: samples[1],
    }),
  })
    .then((response) => {
      if (response.status >= 400 && response.status < 600) {
        throw new Error('服务器离线');
      }
      return response.json();
    })
    .then((res) => {
      result.flag = res.flag;
      const { data } = res;
      result.data['type-1'] = data[0];
      result.data['type-2'] = data[1];
      if (res?.flag === true) {
        result.message = '代码重复度高，识别为克隆代码';
      } else {
        result.message = '代码重复度较低';
      }
      return result;
    })
    .catch((e: Error) => {
      result.message = result?.message ? result?.message : e.message;
      result.error = true;
      return result;
    });
};
