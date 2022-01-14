export interface Result {
  flag: boolean;
  message: string,
  error: boolean
}

const api = process.env.NODE_ENV === "development" ? "" : "https://cloud.ethanloo.cn";
export const detectClone = async (samples: string[]) => {
  if (samples.length < 2) {
    return;
  }
  const result: Result = {
    flag: false,
    message: "",
    error: false
  }
  return await fetch(`${api}/detect`, {
    method: "POST",
    body: JSON.stringify({
      "codeX": samples[0],
      "codeY": samples[1]
    })
  }).then(response => {
    if (response.status >= 400 && response.status < 600) {
      throw new Error("服务器离线");
    }
    return response.json();
  }).then(result => {
    if (result?.data === true) {
      result.flag = true;
      result.message = "代码重复度高，识别为克隆代码"
    } else {
      result.message = "代码重复度较低"
    }
    return result;
  }).catch(e => {
    result.message = `报错原因：${e}`;
    result.error = true;
    return result;
  });
};
