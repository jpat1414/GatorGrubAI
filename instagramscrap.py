from flask import Flask, request, jsonify
2from instaloader import Instaloader
3
4app = Flask(__name__)
5insta_loader = Instaloader()
6
7@app.route('/download_posts', methods=['GET'])
8def download_posts():
9    username = request.args.get('username')
10    if not username:
11        return jsonify({'error': 'Username is required'}), 400
12
13    try:
14        profile = insta_loader.check_profile_id(username)
15        posts = insta_loader.get_posts(profile)
16        post_data = []
17
18        for post in posts:
19            if len(post_data) >= 2:
20                break
21
22            post_info = {
23                'post_url': post.url,
24                'post_caption': post.caption,
25                'post_date': post.date_local,
26                'post_likes': post.likes,
27                'post_comments': post.comments
28            }
29            post_data.append(post_info)
30
31        return jsonify(post_data)
32
33    except Exception as e:
34        return jsonify({'error': str(e)}), 500
35
36if __name__ == '__main__':
37    app.run(debug=True)
