import requests
import time
from multiprocessing import Pool

def send_request(i):
    prompt = f"Request {i}: Give me a short intro to large language models."
    send_time = time.time()

    try:
        response = requests.post(
            "http://localhost:8000/generate",
            json={"prompt": prompt},
            timeout=2400  # Optional: avoids hanging forever
        )
        receive_time = time.time()
        status = response.status_code

        if status != 200:
            return (
                f"[Request {i}] FAILED with status code {status}\n"
                f"Client send time:     {send_time:.6f}\n"
                f"Client receive time:  {receive_time:.6f}\n"
                f"Round-trip time:      {receive_time - send_time:.2f} s\n"
                f"Response: Request failed or timed out.\n"
                + "-"*60 + "\n"
            )

        # If response is OK, try to parse JSON
        try:
            resp_json = response.json()
        except requests.exceptions.JSONDecodeError:
            return (
                f"[Request {i}] FAILED: JSON decode error\n"
                f"Status: {status}\n"
                f"Response Text: {response.text}\n"
                + "-"*60 + "\n"
            )

        # Extract server-side timing if available
        timing = resp_json.get("timing", {})
        server_inference_time = timing.get("inference_duration", None)
        total_latency = timing.get("total_latency", None)

        result = (
            f"[Request {i}]\n"
            f"Client send time:     {send_time:.6f}\n"
            f"Client receive time:  {receive_time:.6f}\n"
            f"Round-trip time:      {receive_time - send_time:.2f} s\n"
        )

        if timing:
            result += (
                f"Server arrival time:  {timing['arrival']:.6f}\n"
                f"Server serve start:   {timing['serving_start']:.6f}\n"
                f"Server inference:     {server_inference_time:.2f} s\n"
                f"Server total latency: {total_latency:.2f} s\n"
            )

        result += f"Response: {resp_json['response']}\n{'-'*60}\n"
        return result

    except Exception as e:
        return (
            f"[Request {i}] EXCEPTION: {str(e)}\n"
            f"Client send time:     {send_time:.6f}\n"
            + "-"*60 + "\n"
        )

if __name__ == "__main__":
    num_procs = 20
    with Pool(processes=num_procs) as pool:
        start_time = time.time()
        for result in pool.imap_unordered(send_request, range(num_procs)):
            print(result)
        end_time = time.time()
        print("-" * 60)
        print(f"Total requests sent: {num_procs}")
        print(f"Total time for {num_procs} requests: {end_time - start_time:.2f} s")
        print(f"Average time per request: {(end_time - start_time) / num_procs:.2f} s")
