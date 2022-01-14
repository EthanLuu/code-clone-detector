import { message } from 'antd';

const api = "http://clone.ethanloo.cn";
export const detectClone = async (samples: string[]) => {
  if (samples.length < 2) {
    return;
  }
  const response = await fetch(`${api}/detect`, {
    method: "POST",
    body: JSON.stringify({
      "codeX": samples[0],
      "codeY": samples[1]
    })
  })
  if (!response.ok) {
    message.error("检测失败");
    return;
  }
  const res = await response.json();
  if (res?.data) {
    message.info("代码类似！")
  } else {
    message.info("代码重复度较低！")
  }
};
