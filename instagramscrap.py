from flask import Flask, request, jsonify
from instaloader import Instaloader

app = Flask(__name__)
insta_loader = Instaloader()

@app.route('/download_posts', methods=['GET'])
def download_posts():
    username = request.args.get('username')
    if not username:
        return jsonify({'error': 'Username is required'}), 400

    try:
        profile = insta_loader.check_profile_id(username)
        posts = insta_loader.get_posts(profile)
        post_data = []

        for post in posts:
            if len(post_data) >= 2:
                break

            post_info = {
                'post_url': post.url,
                'post_caption': post.caption,
                'post_date': post.date_local,
                'post_likes': post.likes,
                'post_comments': post.comments
            }
            post_data.append(post_info)

        return jsonify(post_data)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
