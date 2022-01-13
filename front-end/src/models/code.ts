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
  const data = await response.json();
  console.log(data)
};
