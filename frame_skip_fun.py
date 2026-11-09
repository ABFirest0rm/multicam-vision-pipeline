import _queue
from time import perf_counter
def safe_put_to_queue(q, data, qname=None):
    if not q.full():
        if qname:
            pf = perf_counter()
        q.put_nowait(data)

        if qname:
            print(f"{qname} {(perf_counter() - pf) * 1000:.2f}")
        return True

    else:
        try:
            count = 0
            if qname:
                count += 1
                print(f"{qname} GETTING FRAME {count}")
            q.get_nowait()
            q.put_nowait(data)
            return True
        except _queue.Empty:
            return False